from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime
from collections import Counter
from typing import Any, Dict, List, Optional

from core.config import config
from core.logger import logger


def safe_imports() -> Optional[Dict[str, Any]]:
    """Load optional langchain-related dependencies safely."""
    try:
        from langchain_community.vectorstores import FAISS
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except Exception:
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

        try:
            from langchain_core.documents import Document
        except Exception:
            from langchain.schema import Document  # type: ignore

        try:
            from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # type: ignore
        except Exception:
            OpenAIEmbeddings = None  # type: ignore
            ChatOpenAI = None  # type: ignore

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
        except Exception:
            from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

        try:
            from langchain_core.prompts import PromptTemplate
        except Exception:
            try:
                from langchain.prompts import PromptTemplate  # type: ignore
            except Exception:
                PromptTemplate = None  # type: ignore

        return {
            "FAISS": FAISS,
            "HuggingFaceEmbeddings": HuggingFaceEmbeddings,
            "Document": Document,
            "OpenAIEmbeddings": OpenAIEmbeddings,
            "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
            "ChatOpenAI": ChatOpenAI,
            "PromptTemplate": PromptTemplate,
        }
    except Exception as exc:
        logger.warning(f"缂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ょ紓宥咃躬瀵鎮㈤崗灏栨嫽闁诲酣娼ф竟濠偽ｉ鍓х＝濞达絽鎼瀷濡炪値鍋勯ˇ鍨繆閻㈢绀嬫い鏍ㄦ皑閻も偓闂備礁澹婇崑鍛哄鈧、鎾愁吋婢跺鎷?langchain 闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌ｉ幋锝呅撻柛銈呭閺屾盯顢曢敐鍡欘槬缂佺偓鍎冲锟犲蓟閿濆顫呴柕蹇婃櫇閸旀悂姊哄Ч鍥р偓妤呭磻閹捐埖宕叉繝闈涱儐椤ュ牊绻涢幋婵嗚埞闁告搩鍓熷娲焻閻愯尪瀚板褍澧界槐鎺楁偐閾忣偄纰嶉梺浼欑秮閺€杈╃紦閻ｅ瞼鐭欓悹鍥﹀嫎閸旀垿寮婚弴鐔虹彾妞ゆ牗纰嶉崳瑙勵殽閻愨晛浜鹃梻鍌氬€峰ù鍥х暦閻㈢绐楅柟閭﹀枛閸ㄦ繈鎮橀悙鐢靛彄閻熸瑥瀚峰Σ褰掑箹缁厜鍋撳畷鍥﹀枈濠碉紕鍋戦崐鏍箰閸洖鍨傛繛宸簻绾惧鏌涢弴銊ョ仭闁稿﹤鐖奸悡顐﹀炊瑜庨幑锝夋煕閺傝法效闁哄矉绻濆畷鎺懳旈埀顒佺閹€鏀介柣妯虹仛閺嗏晛鈹戦鎯у幋鐎殿噮鍋婂畷銊︾節閸愩劌浼庨梺璇叉捣閺佸摜娑甸崼鏇炵；闁圭偓鍓氬鈺傘亜閹哄棗浜惧┑鈩冨絻椤兘骞? {exc}")
        return None


class VectorStoreService:
    def __init__(self) -> None:
        self.imports = safe_imports()
        self._lock = threading.Lock()
        self.vs = None
        self.embeddings = None
        self._document_index: List[Any] = []
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._preloaded_vs = None
        self.reload_config()

    def reload_config(self) -> None:
        config.reload()
        data_dir = config.get("storage.data_dir", "data")
        self.cfg = {
            "persist_dir": config.get(
                "rag.persist_dir",
                os.path.join(data_dir, "vector_store"),
            ),
            "embedding_model": config.get(
                "rag.embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            "embedding_provider": config.get("rag.embedding_provider", "huggingface"),
            "api_key": config.get("rag.api_key", ""),
            "base_url": config.get("rag.base_url", "https://api.openai.com/v1"),
            "chunk_size": int(config.get("rag.chunk_size", 600)),
            "chunk_overlap": int(config.get("rag.chunk_overlap", 100)),
            "keyword_search_enabled": bool(config.get("rag.keyword_search_enabled", True)),
            "keyword_weight": float(config.get("rag.keyword_weight", 0.3)),
            "vector_weight": float(config.get("rag.vector_weight", 0.7)),
            "enable_tfidf": bool(config.get("rag.enable_tfidf", True)),
            "enable_dynamic_chunking": bool(config.get("rag.enable_dynamic_chunking", True)),
            "preset_knowledge_enabled": bool(config.get("rag.preset_knowledge_enabled", True)),
            "min_content_length": int(config.get("rag.min_content_length", 50)),
            "pdf_boost": float(config.get("rag.pdf_boost", 0.15)),
            "subject_match_boost": float(config.get("rag.subject_match_boost", 0.15)),
            "query_match_boost": float(config.get("rag.query_match_boost", 0.25)),
            "primary_doc_boost": float(config.get("rag.primary_doc_boost", 0.05)),
            "supplementary_doc_penalty": float(config.get("rag.supplementary_doc_penalty", 0.03)),
            "subject_mismatch_penalty": float(config.get("rag.subject_mismatch_penalty", 0.18)),
            "diversity_threshold": float(config.get("rag.diversity_threshold", 0.3)),
            "hybrid_candidate_multiplier": int(config.get("rag.hybrid_candidate_multiplier", 4)),
            "max_chunks_per_source": int(config.get("rag.max_chunks_per_source", 4)),
            "max_chunks_per_title": int(config.get("rag.max_chunks_per_title", 1)),
            "hf_endpoint": config.get("rag.hf_endpoint", ""),
            "enable_rerank": bool(config.get("rag.enable_rerank", False)),
            "rerank_top_k": int(config.get("rag.rerank_top_k", 8)),
            "excluded_files": list(config.get("rag.excluded_files", [])),
        }
        self._data_dir = os.path.abspath(data_dir)
        self._documents_dir = os.path.abspath(
            config.get("rag.documents_dir", os.path.join(data_dir, "documents"))
        )
        self._url_sources_path = os.path.join(self._data_dir, "rag_url_sources.json")
        self._signature_extensions = {
            ".pdf",
            ".md",
            ".markdown",
            ".txt",
            ".json",
            ".py",
            ".js",
            ".ts",
            ".java",
            ".c",
            ".cpp",
            ".html",
            ".css",
        }
        self._build_meta_path = os.path.join(self.cfg["persist_dir"], "build_meta.json")

    def _normalize_signature_path(self, path: str) -> str:
        return str(path or "").replace("\\", "/").lstrip("./")

    def _iter_local_knowledge_files(self) -> List[str]:
        if not os.path.isdir(self._documents_dir):
            return []

        files: List[str] = []
        for root, _, names in os.walk(self._documents_dir):
            for file_name in names:
                ext = os.path.splitext(file_name)[1].lower()
                if ext not in self._signature_extensions:
                    continue
                file_path = os.path.join(root, file_name)
                if self._should_skip_file(file_path, self._documents_dir):
                    continue
                files.append(file_path)
        return files

    def _normalize_signature_snapshot(
        self, signatures: Optional[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        normalized: Dict[str, Dict[str, float]] = {}
        excluded_files = {
            str(name).strip()
            for name in self.cfg.get("excluded_files", [])
            if str(name).strip()
        }
        skip_names = {
            "kb_chunk_preview.md",
            "rag_testcases.jsonl",
            "rag_testcases_enriched.jsonl",
            "intent_testcases.jsonl",
        }
        for raw_path, raw_meta in (signatures or {}).items():
            if not isinstance(raw_meta, dict):
                continue

            path = self._normalize_signature_path(raw_path)
            file_name = os.path.basename(path)
            if file_name in excluded_files or file_name in skip_names:
                continue
            if path == "rag_url_sources.json":
                normalized[path] = {
                    "mtime": float(raw_meta.get("mtime", 0)),
                    "size": float(raw_meta.get("size", 0)),
                }
                continue

            if not path.startswith("documents/"):
                continue

            ext = os.path.splitext(path)[1].lower()
            if ext not in self._signature_extensions:
                continue

            normalized[path] = {
                "mtime": float(raw_meta.get("mtime", 0)),
                "size": float(raw_meta.get("size", 0)),
            }
        return normalized

    def _collect_document_signatures(self) -> Dict[str, Dict[str, float]]:
        signatures: Dict[str, Dict[str, float]] = {}
        persist_dir = os.path.abspath(self.cfg["persist_dir"])

        for file_path in self._iter_local_knowledge_files():
            if os.path.abspath(file_path).startswith(persist_dir):
                continue
            try:
                stat = os.stat(file_path)
                relative = self._normalize_signature_path(
                    os.path.relpath(file_path, self._data_dir)
                )
                signatures[relative] = {"mtime": stat.st_mtime, "size": stat.st_size}
            except OSError as exc:
                logger.warning(f"Failed to collect signature for {file_path}: {exc}")

        if os.path.exists(self._url_sources_path):
            try:
                stat = os.stat(self._url_sources_path)
                signatures["rag_url_sources.json"] = {
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                }
            except OSError as exc:
                logger.warning(f"Failed to collect URL source signature: {exc}")
        return signatures

        for root, dirs, files in os.walk(data_dir):
            abs_root = os.path.abspath(root)
            if abs_root.startswith(persist_dir):
                dirs[:] = []
                continue
            for file_name in files:
                ext = os.path.splitext(file_name)[1].lower()
                if ext not in self._signature_extensions:
                    continue
                file_path = os.path.join(root, file_name)
                try:
                    stat = os.stat(file_path)
                    relative = os.path.relpath(file_path, data_dir)
                    signatures[relative] = {"mtime": stat.st_mtime, "size": stat.st_size}
                except OSError as exc:
                    logger.warning(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧湱鈧懓瀚崳纾嬨亹閹烘垹鍊炲銈嗗坊閸嬫挻娼诲┑瀣拺闂侇偆鍋涢懟顖涙櫠閹绢喗鐓涢悘鐐插⒔閵嗘帡鏌嶈閸撱劎绱為崱娑樼；闁告侗鍘鹃弳锔锯偓鍏夊亾闁告洦鍓涢崢鎼佹煟韫囨洖浠滃褑妫勭叅閻庢稒菧娴滄粓鏌曡箛濠傚⒉缂佲偓鐎ｎ喗鐓涘ù锝囨嚀婵秶鈧娲栧畷顒勫煝鎼粹垾鐔煎箒瀹ュ洤浠辨慨濠傤煼瀹曟帒鈻庨幇顔哄仒濠电姰鍨婚幊鎾绘偋韫囨梻顩茬紒瀣氨閺嬪酣鏌熼柇锕€鏋ゆい鏃€娲熷娲焻閻愯尪瀚板褎鎸抽弻锛勪沪缁洖浜剧€规洖娲﹀▓鏇㈡煟鎼搭垳绉甸柛鎾寸洴閹線宕奸妷锕€鈧敻鎮峰▎蹇擃仾缂佲偓閸愵喗鍋ㄦい鏍ㄧ☉濞搭噣鏌ㄥ┑鍫濅粶闁宠鍨归埀顒婄到閻忔岸寮查敐澶嬧拺缂備焦锚閻忋儲淇婇锝囨噯闁愁亝鎮傚濠氬磼濮橆兘鍋撻幖浣哥９濡炲瀛╅鑺ユ叏濮楀棗澧婚柣鎺旀櫕閹茬顓奸崼顫綍濠电姴鐥夐弶搴撳亾閺囥垹绠犻幖杈剧到椤ユ氨鎲搁悧鍫濈瑲闁绘挾濮撮埞鎴︻敊閽樺顫岄梺閫炲苯澧繛鑼枎閻ｇ兘鎮烽幍铏杸闁诲函缍嗛崑鍡涘储娴犲鈷戦梻鍫熺〒缁犵偤鏌涢埡浣割伃闁诡喚鍋ゅ畷婊嗩槾缁炬儳銈搁弻锝呂熼幐搴ｅ涧闂佹眹鍔嶉崹鍧楀蓟閻旂厧绠甸柟鐑樻尭閺嗗牓姊虹拠鈥崇仭婵☆偄鍟穱濠囧箹娴ｈ倽銊ф喐濠靛牊顫曟い鏇楀亾婵﹦绮幏鍛村川婵犲倹娈橀梻浣告啞濮婄懓煤閺嶎剛浜藉┑鐘垫暩婵潙煤閿曞倸纾婚柛鏇ㄥ灡閻撴洟鏌嶉埡浣告殶闁愁垱娲熼弻锝夊箻鐎靛憡鍣梺闈涙搐鐎氭澘顕ｆ禒瀣╃憸澶娢ｉ崼婵愭富闁靛牆鍟崝姘亜閹存繃宸濇俊鍙夊姍楠炴帒螖閳ь剚鍎梻渚€娼ч¨鈧紒鐘冲灦缁傛帡骞橀瑙ｆ嫼? {file_path}, 闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻锝夊箣閿濆憛鎾绘煕婵犲倹鍋ラ柡灞诲姂瀵噣宕奸悢鍛婎唶闂備胶顭堥鍡涘箰閸撗冨灊妞ゆ挾鍋愬Σ鍫熶繆椤栨繍鍤欐い锔规櫊濮婅櫣绮欏▎鎯у妧缂備胶绮换鍫濐嚕婵犳艾鍗抽柣鏃囨椤旀洟姊虹紒妯哄Е闁告挻宀搁幃鐢告晝閸屾稈鎷? {exc}")
        return signatures

    def _load_build_metadata(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self._build_meta_path):
            return None
        try:
            with open(self._build_meta_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as exc:
            logger.warning(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾剧懓顪冪€ｎ亝鎹ｉ柣顓炴閵嗘帒顫濋敐鍛婵°倗濮烽崑娑⑺囬悽绋垮瀭濡わ絽鍟粻娑樏归敐鍛础缂佸鑳剁槐鎾诲磼濮橆兘鍋撻悜鑺ュ€块柨鏇氱劍閹冲矂姊绘担鍛婂暈闁荤喆鍎抽幑銏狀潨閳ь剙顕ｆ繝姘亜缁炬媽椴搁弲顒勬⒑閹稿海绠撴繛璇х到鏁堟俊銈呮噺閸嬧剝绻濇繝鍌涘櫣妞わ絽銈搁幃浠嬵敍濞戞ɑ璇為梺璇″枟閻燂妇鎹㈠┑瀣闁挎棁娉曢惄搴ㄦ⒒娴ｇ儤鍤€闁圭⒈鍋勮灋婵°倓鑳堕々鎻捨旈敐鍛殲闁绘挾鍠栭弻锝夊棘閹稿孩鍠愰梺纭呮珪濞叉粎妲愰幒妤佸亹闁惧浚鍋勭壕鎶芥倵濞堝灝鏋涙い顓犲厴楠炲啴濮€閵堝懎绐涙繝鐢靛Т鐎氼喛鍊撮梻鍌氬€风粈渚€骞夐敓鐘茬闁糕剝绋戝婵囥亜閺冨倻鎽傞柛鐔锋噺缁绘繈鎮介棃娴躲儵鏌℃担瑙勫€愮€规洘鍨甸埥澶愬閻樼數鏋冮柣搴″帨閸嬫捇鏌涢弴銊ヤ簽闁挎稒鐟ラ埞鎴︽偐缂佹ɑ閿銈嗗灥濞层倝鎮鹃崹顐ょ懝闁逞屽墴瀵鍨鹃幇浣告倯闁硅偐琛ラ埀顒冨皺閻栭亶姊绘担渚劸妞ゆ垵妫濋獮鎰板箹娴ｅ摜鍙€婵犮垼娉涜墝闁哄绉归弻褑绠涢敐鍛盎闂佽绻嗛弲娑㈡箒濠电姴锕ょ€氼參宕愰幇顓滀簻闁靛繆妲勯懓鍨攽椤旂懓浜鹃梻浣瑰缁嬫垹鈧凹鍓熼悰顕€宕奸悢铏圭槇缂佸墽澧楄摫妞ゎ偄锕弻娑㈠Ω閵壯嶇礊闂佸憡甯楃敮锟犮€佸☉姗嗘僵閺夊牄鍔嶉鏇㈡煟鎼粹€冲辅闁稿鎹囬弻宥堫檨闁告挾鍠栧畷娲焵椤掍降浜滈柟鍝勭Ф鐠愪即鏌涢悢椋庣闁哄本鐩弫鎰償閳╁啰褰庨柣搴ゎ潐濞叉﹢宕归崜浣瑰床婵犻潧顑呴悙濠勬喐韫囨稒鍋╂繛宸簼閸嬧剝绻濇繝鍌氼仾闁绘挶鍎甸弻锝夋晲閸パ冨箣闂佽鍠掗崜婵嬪箚閺傝鐔虹磼濡粯绶梻鍌氬€风粈浣圭珶婵犲洤纾婚柛娑卞灣閻瑩鏌熺€电浠фい鈺傜叀閺岋絽螖閳ь剟鎮ц箛鏇炲К闁逞屽墴濮婂搫效閸パ呬紙濠电偘鍖犻崱蹇撳緮闂傚倸鍊烽悞锕傚箖閸洖绀夌€广儱鎷嬪〒濠氭煢濡警妲洪柡鍡畵閺岋繝宕堕妷銉т痪闂佹悶鍊曢崯鎾蓟瀹ュ棙濮滈柟娈垮櫘濡差噣姊虹拠鑼闁挎洦浜滈～蹇撁洪鍕獩婵犵數濮撮崯顐︽儊閻戣姤鈷戦柛娑橈攻婢跺嫰鏌涚€ｎ亝鍤囬柟? {exc}")
            return None

    def _save_build_metadata(self, signatures: Dict[str, Dict[str, float]]) -> None:
        try:
            os.makedirs(os.path.dirname(self._build_meta_path), exist_ok=True)
            with open(self._build_meta_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "built_at": datetime.now().isoformat(),
                        "signature_scope": "knowledge_sources_v2",
                        "documents": signatures,
                    },
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as exc:
            logger.warning(f"濠电姷鏁告慨鐑藉极閸涘﹥鍙忛柣鎴ｆ閺嬩線鏌熼梻瀵割槮缁炬儳娼￠弻鐔衡偓鐢殿焾瀛濈紓浣界堪閸婃繈寮婚敃鈧灒濞撴凹鍨遍敍鍡椻攽閻愬弶鈻曞ù婊勭箞瀵彃顭ㄩ崼鐔哄幗闂侀€涘嵆濞佳勬櫠椤曗偓楠炴牠寮堕幋顖濆惈濠殿喖锕ㄥ▍锝囧垝濞嗘垶宕夐柕濞у倻绀冮梻鍌欒兌椤牓顢栭崶顒€纾婚柕鍫濇噽閺嗭箓鏌熸潏鍓х暠閸ュ瓨绻濋姀锝嗙【妞ゆ垵娲畷銏ゎ敂閸涱垳鐦堥梺姹囧灲濞佳勭瑜旈弻娑氣偓锝庡亝鐏忣參鏌嶇拠鑼х€规洏鍔庨埀顒佺⊕閿氬ù鐙€鍘奸埞鎴︽倷閸欏妫￠梺鑽ゅ暀閸ャ劌浠奸梺姹囧灩閹诧繝鎮″☉姘ｅ亾閸忓浜鹃梺閫炲苯澧寸€规洑鍗抽獮鍥敆婢跺苯濮洪梻濠庡亜濞诧妇绮欓幋鐘电焼闁割偁鍨洪崰鎰扮叓閸ャ劎鈽夋慨瑙勭叀閺岋絽螣閼姐們鍋為梺琛″亾濞寸姴顑嗛悡鐔兼煙闁箑澧紒鐙欏洦鐓曢柨婵嗘搐閸樻挳鏌＄仦鍓р姇缂佺粯绻堝畷姗€顢旈崱娆愭緰闂傚倷鑳剁划顖涚瑹濡ゅ懎闂柨婵嗘媼閸ゆ洖霉閻樺樊鍎涢柡浣稿暣閺岋綁寮崶銊ㄥ悅婵犮垼顫夊ú鏍煘閹达附鍊烽柡澶嬪灩娴犵顪冮妶鍛闁告濞婇悰顔锯偓锝庡枟閸嬫劙寮堕崼姘珔濠⒀呭亾娣囧﹪鎮欓鍕ㄥ亾閺嶎厼绠伴柛顐ｆ礀閸ㄥ倹绻涘顔荤凹闁稿鍊块弻锟犲炊閳轰焦鐎虹紓浣筋嚙濡瑩濡甸崟顖氬唨闁靛ě鍛帓闂佹眹鍩勯崹杈╃矙閹捐鐒垫い鎺戝枤濞兼劖绻涢崣澶屽⒌闁诡垰顦扮缓鐣岀矙鐠恒劌濮︽俊鐐€栫敮濠勬閿熺姴鐤煫鍥ㄧ⊕閻撴洟鏌ㄥ┑鍡涱€楁繛鍛Ф閳ь剝顫夊ú姗€宕归崜浣瑰床婵犻潧顑呴悙濠勬喐韫囨稒鍋╂繛宸簼閸嬧剝绻濇繝鍌氼仾闁绘挶鍎甸弻锝夋晲閸パ冨箣闂佽鍠掗崜婵嬪箚閺傝鐔虹磼濡粯绶梻鍌氬€风粈浣圭珶婵犲洤纾婚柛娑卞灣閻瑩鏌熺€电浠фい鈺傜叀閺岋絽螖閳ь剟鎮ц箛鏇炲К闁逞屽墴濮婂搫效閸パ呬紙濠电偘鍖犻崱蹇撳緮闂傚倸鍊烽悞锕傚箖閸洖绀夌€广儱鎷嬪〒濠氭煢濡警妲洪柡鍡畵閺岋繝宕堕妷銉т痪闂佹悶鍊曢崯鎾蓟瀹ュ棙濮滈柟娈垮櫘濡差噣姊虹拠鑼闁挎洦浜滈～蹇撁洪鍕獩婵犵數濮撮崯顐︽儊閻戣姤鈷戦柛娑橈攻婢跺嫰鏌涚€ｎ亝鍤囬柟? {exc}")

    def _documents_changed(
        self, previous: Dict[str, Any], current: Dict[str, Dict[str, float]]
    ) -> bool:
        previous = self._normalize_signature_snapshot(previous)
        current = self._normalize_signature_snapshot(current)
        if len(previous) != len(current):
            return True
        for path, meta in current.items():
            old = previous.get(path)
            if not old:
                return True
            if round(old.get("mtime", 0), 3) != round(meta.get("mtime", 0), 3):
                return True
            if old.get("size") != meta.get("size"):
                return True
        return False

    def _ensure_embeddings(self) -> None:
        if self.embeddings is not None or self.imports is None:
            return

        provider = str(self.cfg.get("embedding_provider", "huggingface")).lower()
        api_key = (
            self.cfg.get("api_key")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
        )
        model_name = self.cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        base_url = self.cfg.get("base_url")
        hf_endpoint = self.cfg.get("hf_endpoint") or os.getenv("HF_ENDPOINT")

        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint

        if provider == "dashscope" and api_key:
            try:
                from langchain_community.embeddings import DashScopeEmbeddings

                logger.info(f"濠电姷鏁告慨鐑藉极閸涘﹥鍙忛柣鎴ｆ閺嬩線鏌熼梻瀵割槮缁炬儳顭烽弻锝呂熷▎鎯ф缂備胶濮撮悘姘跺Φ閸曨喚鐤€闁圭偓鎯屽Λ銈囩磽娴ｆ彃浜炬繝鐢靛Т濞诧箓鎮￠崘顏呭枑婵犲﹤鐗嗙粈鍫熸叏濡潡鍝虹€规洖寮剁换娑㈠箣閻愬灚鍣х紓浣稿閸嬨倝骞冨Δ鍛櫜閹肩补鈧尙鏁栨俊鐐€х紓姘跺础閹惰棄绠?DashScope 闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾剧懓顪冪€ｎ亝鎹ｉ柣顓炴閵嗘帒顫濋敐鍛婵°倗濮烽崑娑⑺囬悽绋挎瀬鐎广儱顦粈瀣亜閹板墎鎮奸柍褜鍓氭繛濠囧箖濡ゅ懐宓侀柛顭戝枛婵骸顪冮妶蹇撶槣闁搞劏娉涢悾鐤亹閹烘垹锛滈梺缁樺姇椤曨參宕㈤柆宥嗗仭婵犲﹤鍟撮崣鍕煏閸℃鏆ｇ€规洏鍔庨埀顒佺⊕鑿ら柟鐤嚙椤啴濡堕崱娆忊拡闂佺顑囬崑銈夊箖閿熺姴鍗抽柣鏃囨閻﹀牓姊婚崒姘卞缂佸甯¤棢婵犲﹤鐗婇悡鐔哥節闂堟稒锛嶉柛搴㈡閺屸€崇暆鐎ｎ剛袦閻庢鍣崳锝呯暦閹烘埈娼╂い鎴ｆ硶鐢盯姊婚崒娆愮グ妞ゆ泦鍛床闁归偊鍠栭ˉ姘亜閹惧崬鐏╅柣鎺戠仛閵囧嫰骞掗幋婵囩亾濠电偛鍚嬮崝鏍崲濞戙垹鐭楀璺鸿嫰閳綊鏌ｉ幘鍗炩偓婵嬪蓟閿濆憘鐔封枎閹勵唲缂? {model_name}")
                self.embeddings = DashScopeEmbeddings(
                    model=model_name,
                    dashscope_api_key=api_key,
                )
                return
            except Exception as exc:
                logger.warning(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾剧懓顪冪€ｎ亝鎹ｉ柣顓炴閵嗘帒顫濋敐鍛婵°倗濮烽崑娑⑺囬悽绋挎瀬鐎广儱顦粈瀣亜閹哄秶鍔嶆い鏂挎处缁绘繂顕ラ柨瀣凡闁逞屽墯濞茬喎鐣烽鐑嗘晬婵椴稿▓楣冩⒑閸濆嫯顫﹂柛搴㈢叀閹繝寮撮悩顐壕闁稿繐顦禍楣冩⒑闁偛鑻晶瀵糕偓瑙勬礃濞茬喖鐛€ｎ喗鍋愰柣銏☆問濡叉悂姊绘担鍛婃儓缂佸绶氬畷銏＄鐎ｎ偄浜楀┑鐐叉閹稿鎮￠悢闀愮箚闁靛牆鍊告禍楣冩⒑缂佹﹩娈旈柨鏇畱閳诲酣濮€閵堝棗娈ゅ銈嗗笒閸婂鏁嶅鍫熲拺?DashScope 闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾剧懓顪冪€ｎ亝鎹ｉ柣顓炴閵嗘帒顫濋敐鍛婵°倗濮烽崑娑⑺囬悽绋挎瀬鐎广儱顦粈瀣亜閹板墎鎮奸柍褜鍓氭繛濠囧箖濡ゅ懐宓侀柛顭戝枛婵骸顪冮妶蹇撶槣闁搞劏娉涢悾鐤亹閹烘垹锛滈梺缁樺姇椤曨參宕㈤柆宥嗗仭婵犲﹤鍟撮崣鍕煏閸℃鏆ｇ€规洏鍔庨埀顒佺⊕鑿ら柟鐤嚙椤啴濡堕崱娆忊拡闂佺顑囬崑銈夊箖閿熺姴鍗抽柣鏃囨閻﹀牓姊婚崒姘卞缂佸甯¤棢婵犲﹤鐗婇悡鐔哥節闂堟稒锛嶉柛搴㈡閺屸€崇暆鐎ｎ剛袦閻庢鍣崳锝呯暦閹烘埈娼╂い鎴ｆ硶鐢盯姊婚崒娆愮グ妞ゆ泦鍛床闁归偊鍠栭ˉ姘亜閹惧崬鐏╅柣鎺戠仛閵囧嫰骞掗幋婵囩亾濠电偛鍚嬮崝鏍崲濞戙垹鐭楀璺鸿嫰閳綊鏌ｉ幘鍗炩偓婵嬪蓟閿濆憘鐔封枎閹勵唲缂備胶鍋撻崕鎶解€﹂悜钘夎摕闁绘梻鍘х粈鍫㈡喐韫囨洘鏆滄繛鎴欏灪閻撶喖鏌熼幆褍鑸归柛鏃撶畵閺岀喖顢欑粵瀣杹闂佽桨鐒﹂崝鏍箚閺傚簱鏀介柛顐ゅ枑濞咃妇绱撻崒姘偓鐑芥倿閿曞倸绀夋俊銈呮噹绾惧鏌曢崼婵愭Ц闁藉啰鍠栭弻宥夊煛娴ｅ憡娈紓浣稿閸嬫盯鍩為幋锔藉亹闁圭粯甯╂禒濂告⒑閸愭彃妲婚柣妤佹尭椤繘鎼归崷顓犵厯濠电偛妫欓崕鎶藉礈閺夋娓婚柕鍫濈箻濡绢喖鈹戦悙璇ч練闁活參绠栧铏圭矙鐠恒劎浼囬梺绋款儑閸嬨倝宕哄☉銏犵闁圭偨鍔岀紞濠囧极閹版澘鐐婇柕濞垮劜閻ｎ剚淇婇悙顏勨偓鏍哄Ο鐓庡灊閹艰揪绲鹃～鏇㈡煙閻戞ɑ鐓涢柛瀣崌閺佹劖鎯旈垾鑼泿闂備浇顕х换鎴﹀箰閹惰棄钃熼柣鏃堫棑閺嗭箓鏌涢妷鎴濆娴滎亝淇婇悙顏勨偓鎴﹀垂濞差亝鍎庢い鏍仜缁犳牗绻涢崱妯诲碍缂佺嫏鍥ㄧ厵閻庢稒顭囩粻鎾淬亜閵夈儺妯€婵﹤顭峰畷鎺戔枎閹寸姷宕叉繝鐢靛仒閸栫娀宕楅悩鍨殌妞ゎ偅绻堥、妤佸緞鐏炶姤鏆╅梻鍌欑窔閳ь剛鍋涢懟顖涙櫠閹绢喗鐓涚€光偓鐎ｎ剛袦濡ょ姷鍋炵敮锟犵嵁鐎ｎ噮鏁嶆繛鎴炵懄濮ｅ海绱撻崒姘偓鐑芥嚄閸撲礁鍨濇い鏍ㄧ矋瀹曟煡鏌涢锝嗗剷婵炴垯鍨圭粈鍐煏婵炲灝鐏い顐㈢Ч濮婃椽宕ㄦ繝鍌氼潔閻熸粍婢橀崯鎾晲閻愮儤鍋愮€规洖娲ㄩ惁鍫㈢磼閸撗冾暭閼裤倝鏌涚€ｎ偅宕岀€规洘锕㈤幊鐘活敆閸曨厼绀堥梻鍌氬€峰ù鍥敋閺嶎厼绐楅柡宥庡幖绾惧綊鏌涢…鎴濇珮闁搞倖娲熼幃褰掑炊閵娧佸仦濠碘剝褰冮悧鎾诲蓟閺囷紕鐤€闁哄洨鍎愬鏃堟⒑闂堟丹娑㈠礋椤愶絿鈧箖鏌ｆ惔锛勭暛闁稿骸宕灋婵炴埈娼块埀顒€鎳橀幃婊堟嚍閵夈垺瀚兼俊鐐€栧濠氬磻閹惧墎纾煎璺烘湰閺嗩剟鏌熼鍡欑瘈鐎规洘锕㈤、鏃堝川椤撶喓鈧偓绻濋悽闈浶㈤柨鏇樺€濋幃褔宕卞▎鎴犵劶闂佺鏈粙鎺楀矗韫囨柧绻嗛柣鎰邦杺閸ゆ瑥霉濠婂牏鐣洪柟? {exc}")

        if provider != "huggingface" and api_key and self.imports.get("OpenAIEmbeddings"):
            OpenAIEmbeddings = self.imports["OpenAIEmbeddings"]
            for kwargs in (
                {
                    "model": model_name,
                    "api_key": api_key,
                    "base_url": base_url,
                    # 向量化接口单次上限 32 条，超出会返回 413
                    "chunk_size": 32,
                },
                {
                    "model": model_name,
                    "openai_api_key": api_key,
                    "openai_api_base": base_url,
                    "chunk_size": 32,
                },
            ):
                try:
                    logger.info(f"濠电姷鏁告慨鐑藉极閸涘﹥鍙忛柣鎴ｆ閺嬩線鏌熼梻瀵割槮缁炬儳顭烽弻锝呂熷▎鎯ф缂備胶濮撮悘姘跺Φ閸曨喚鐤€闁圭偓鎯屽Λ銈囩磽娴ｆ彃浜炬繝鐢靛Т濞诧箓鎮￠崘顏呭枑婵犲﹤鐗嗙粈鍫熸叏濡潡鍝虹€规洖寮剁换娑㈠箣閻愬灚鍣х紓浣稿閸嬨倝骞冨Δ鍛櫜閹肩补鈧尙鏁栨俊鐐€х紓姘跺础閹惰棄绠栨俊銈傚亾闁崇粯鎹囧畷褰掝敊閻ｅ苯钂嬮梻鍌欒兌椤牓顢栭崨瀛樻櫇闁靛濡囬埞宥呪攽閻樺弶鎼愰梺鍗炴喘閺岋繝宕堕埡浣风捕闂佸吋婢橀悘婵嬪煘閹达附鍊烽柛娆忣樈濡繝鏌﹂崘顔绘喚婵﹨娅ｉ幉妤呭Χ閸涱亜浜炬繝闈涙－濞兼牜绱撴担鑲℃垶鍒婇幘顔界厽闁瑰浼濋鍕ㄦ灁閻忕偟鐡斿〒濠氭煏閸繃顥炲璺哄閺屾稑螣閸︻厾鐓撻悗瑙勬礃绾板秶鎹㈠┑瀣倞闁肩鐏氬▍宥夋⒒娴ｈ櫣甯涢柡灞诲姂閵嗗啯鎯旈妸銉ユ優闂侀潧鐗嗛ˇ浼村煕閹寸姷纾奸悗锝庡幗绾爼鏌ｈ箛娑楁喚闁哄本绋撻埀顒婄秵閸撴瑦绂掗柆宥嗙厵妞ゆ洖妫涚弧鈧悗娈垮枟閹告娊骞冮姀銏㈢煓闁割煈鍠曠槐鐔封攽閻樻剚鍟忛柛鐘愁殜閵嗗啴宕ㄩ鍥ㄧ☉铻栭柛娑卞灠缁侊箓姊洪崫鍕枆闁告ü绮欓幃锟犳偄閼测晛褰勯梺鎼炲劘閸斿秹鍩涢幇鐗堝€堕煫鍥ュ劤閻ｇ敻鏌″畝瀣？濞寸媴绠撳畷婊嗩槼闁告帗鐩鐑樻姜閹殿喖濡介梺鍛婃⒐閻熴儵鎮鹃悜钘壩╅柍鍝勶攻閺呪晠姊洪懞銉冾亪藝椤栨稓鐝剁€广儱顦伴埛鎺懨归敐鍥剁劸缂傚秵鐗滅槐鎺擃潙閺嵮勭亾缂備浇浜崑鐐电箔閻旂厧鐒垫い鎺嗗亾妞ゆ洩绲块幏鐘差啅椤斿吋鍊┑鐘灱濞夋稖鐧岄梺闈浥堥弲婊堟偂韫囨搩鐔嗛悹鍥ｂ偓宕囦哗缂備礁顦晶搴ｅ垝閸喓绡€闁搞儯鍔夐幏? {model_name}")
                    self.embeddings = OpenAIEmbeddings(**kwargs)
                    return
                except Exception:
                    continue

        if provider != "huggingface" and not api_key:
            logger.error(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾剧懓顪冪€ｎ亝鎹ｉ柣顓炴閵嗘帒顫濋敐鍛婵°倗濮烽崑娑⑺囬悽绋挎瀬鐎广儱顦粈瀣亜閹板墎鎮奸柍褜鍓氭繛濠囧箖濡ゅ懐宓侀柛顭戝枛婵骸顪冮妶蹇撶槣闁搞劏娉涢悾鐤亹閹烘垹锛滈梺缁樺姇椤曨參宕㈤柆宥嗗仭婵犲﹤鍟撮崣鍕煏閸℃鏆ｇ€规洏鍔庨埀顒佺⊕鑿ら柟鐤嚙椤啴濡堕崱娆忣潷缂備礁顑嗛幐鍓у垝婵犳艾绀堢憸澶愬磻閹剧粯鍋￠柡澶嬵儥濡矂姊洪崫銉バｆ繛鍙夌矒閹崇偞娼忛妸褏鐦堝┑顔斤供閸樺ジ鍩€椤掆偓閻忔岸骞堥妸銉建闁糕剝顨呴埛鎺戔攽閳ュ啿绾ф俊鐐扮矙瀵顓奸崼顐ｎ€囬梻浣告啞閹稿鎮疯閸掓帞鎷犲顔兼倯婵犮垼娉涢敃锕傛晬閻斿吋鈷掑〒姘搐婢ь喚绱掓径濠庡殶缂侇喖顭烽幃鈺冩嫚閼碱剦鍟囬梻浣告贡閳峰牓宕㈡ィ鍐ㄥ偍婵犲﹤鎳愮壕鐣屸偓骞垮劚濡稒鏅堕悽鍛婄厸閻忕偟鍋撶粈瀣偓瑙勬礈閸樠囧煘閹达箑绠涙い鎾筹紡閸ャ劉鎷虹紓鍌欑劍钃遍悘蹇庡嵆閺岋綁鎮㈠┑鍡樺櫤妞ゎ偅娲熼弻鐔煎箚瑜忛幗鐘测攽椤栨稒缍戦柍瑙勫灴椤㈡瑩鎸婃径灞藉Ъ濠电偛鐡ㄧ划宥囨崲閸儳宓侀柛顐犲劚鎯熼梺瀹犳〃閼冲爼藝閵娿儺娓婚柕鍫濇噽缁犵増淇婇锝囨创闁轰礁绉撮…銊╁醇閻斿弶瀚奸梻浣告啞缁嬫垿鏁冮妷褌鐒婃い鎰堕檮閻撶喖鏌熼搹鐟颁户闁伙絿鏁搁埀顒侇問閸犳绻涙繝鍥モ偓浣肝旈崨顓狅紲濠德板€愰崑鎾趁?{provider} 缂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚悢鍏尖拻閻庨潧澹婂Σ顔剧磼閻愵剙鍔ょ紓宥咃躬瀵鎮㈤崗灏栨嫽闁诲酣娼ф竟濠偽ｉ鍓х＝濞达絽鎼瀷濡炪値鍋勯ˇ鍨繆閻㈢绀嬫い鏍ㄦ皑閻も偓闂備礁澹婇崑鍛哄鈧、鎾愁吋婢跺鎷?API Key")
            return

        try:
            HuggingFaceEmbeddings = self.imports["HuggingFaceEmbeddings"]
            logger.info(f"濠电姷鏁告慨鐑藉极閸涘﹥鍙忛柣鎴ｆ閺嬩線鏌熼梻瀵割槮缁炬儳顭烽弻锝呂熷▎鎯ф缂備胶濮撮悘姘跺Φ閸曨喚鐤€闁圭偓鎯屽Λ銈囩磽娴ｆ彃浜炬繝鐢靛Т濞诧箓鎮￠崘顏呭枑婵犲﹤鐗嗙粈鍫熸叏濡潡鍝虹€规洖寮剁换娑㈠箣閻愬灚鍣х紓浣稿閸嬨倝骞冨Δ鍛櫜閹肩补鈧尙鏁栨俊鐐€х紓姘跺础閹惰棄绠栨俊銈傚亾闁崇粯鎹囧畷褰掝敊閻ｅ苯钂嬮梻鍌欒兌椤牓顢栭崨瀛樻櫇闁靛濡囬埞宥呪攽閻樺弶澶勯柛銈咁儔濮婂宕奸悢鎭掆偓鎺旂磼閹邦喖浠︾紒缁樼箞閹粙妫冨ù韬插灲閺屾稒绻濋崘鈺婃健闂侀潧娲﹂崝鏍ь嚗閸曨厸鍋撻敐鍌涙珖缂佺姵宀稿铏圭磼濡搫袝婵炲瓨绮嶇划鎾诲春閳ь剚銇勯幒宥堝厡濠⒀屼簼閵囧嫰寮捄銊ь唹缂備胶绮换鍌烇綖濠靛鏁嗛柛宀嬪楠炪垽姊婚崒娆愮グ妞ゎ偄顦靛畷鏇㈠箮缁涘鏅銈嗘尵婵瓨鎱ㄩ鍕厓鐟滄粓宕滈悢濂夋綎婵炲樊浜濋悞濠氭煟閹邦垰钄奸悗姘緲椤儻顦查柣掳鍔戦妴鍐Ψ閳哄倸鈧兘寮堕崼鐔风闁逞屽墯鐎笛囧Φ閸曨垰顫呴柨娑樺閸ｄ即姊虹涵鍛彧缂佽鐗嗚灋闁告劦鍠楅崵瀣煕椤垵浜炴い锔芥緲閳规垿顢欑涵鐑界反濠电偛鎷戠徊浠嬪煡婢舵劖鍋ㄧ紒瀣仢缁愭稑顪冮妶鍡樼叆闁告艾顑夐弻銊╊敇閻樼數锛濋梺绋挎湰閻熴劑宕楃仦瑙ｆ斀妞ゆ梻鍋撻弳顒傗偓瑙勬礀缂嶅﹪骞冮悾宀€鐭欓悹鎭掑妼楠炴帡鏌ｆ惔锛勭暛闁稿酣浜惰棟妞ゆ牗纰嶇痪顖炴⒒閸屾瑨鍏岀痪顓炵埣瀹曟粌鈹戦崰銏犳惈椤劑宕ㄩ娆戠憹婵犳鍠楅敃鈺呭礈閿曗偓閵嗘帗绻濆顓犲帾闂佸壊鍋呯换鍐夐悩宸唵鐟滄粓宕滈悢鐓庤摕闁哄洢鍨归悙濠囨煏婵炑冨閸ㄩ亶鏌ｆ惔銈庢綈婵炲弶锕㈠畷褰掑锤濡ゅ啫绁﹀┑鈽嗗灥濞咃綁寮抽敃鍌涚厱闁靛绲芥俊鑺ャ亜閺冣偓濞茬喎顫忕紒妯诲閻熸瑥瀚禒鈺呮⒑缁嬫寧鍞夌€规洟娼ч銉︾節閸ャ劌娈熼梺闈涳紡鐏為箖鏁? {model_name}")
            self.embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"local_files_only": True})
        except Exception as exc:
            logger.error(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾剧懓顪冪€ｎ亝鎹ｉ柣顓炴閵嗘帒顫濋敐鍛婵°倗濮烽崑娑⑺囬悽绋垮瀭濡わ絽鍟粻娑樏归敐鍛础缂佸鑳剁槐鎾诲磼濮橆兘鍋撻悜鑺ュ€块柨鏇氱劍閹冲矂姊绘担鍛婂暈闁荤喆鍎抽幑銏狀潨閳ь剙顕ｆ繝姘亜缁炬媽椴搁弲顒勬⒑閹稿海绠撴繛璇х到鏁堟俊銈呮噺閸嬧剝绻濇繝鍌涘櫣妞わ絽銈搁幃浠嬵敍濞戞ɑ璇為梺璇″枟閻燂妇鎹㈠┑瀣闁挎棁娉曢惄搴ㄦ⒒娴ｇ儤鍤€闁圭⒈鍋勮灋婵°倓鑳堕々鎻捨旈敐鍛殲闁绘挾鍠栭獮鏍庨鈧悘鍗烆熆鐟欏媶鎴﹀Φ閸曨垱鏅滈柛顭戝枛缁侇喗绻濈喊澶岀？闁轰浇顕ч悾鐑芥偄绾拌鲸鏅┑顔斤耿绾悂宕ú顏呪拻濞达綀娅ｉ妴濠囨煕閹惧绠炲┑锛勬暬閹瑧鈧潧鎽滆ぐ楣冩⒑閸濆嫭宸濋柛鐘虫尵瀵囧焵椤掑嫭鈷戞慨鐟版搐閻忓弶绻涙担鍐叉閸欐挳姊婚崒娆掑厡妞ゎ厼鐗撻、鏍幢濞戞顔夐梺鎼炲労閸撴瑩鎷戦悢鍏肩厽闁哄啫娲﹂鐘电棯閹佸仮闁哄本娲濈粻娑㈠即閻愭劧缍侀弻锝堢疀閹捐泛鈪甸梺鍝勭焿缂嶄線骞冮姀銈呬紶闁靛绠戝▓蹇曠磽閸屾瑨鍏屽┑顔炬暬楠炴垿宕惰閸ゆ洖鈹戦悩宕囶暡闁稿瀚伴弻娑樷槈濮楀牆濮涚紓浣割槸閻栫厧顫忓ú顏勫窛濠电姴鍟惁鐑芥⒑閸涘﹣绶遍柛鐘愁殜椤㈡瑨绠涢幘顖涙杸闂佺粯锚閻忔岸寮抽埡鍛厱閻庯綆鍋嗗ú瀛橆殽閻愯宸ラ柣锝嗙箞瀹曠喖顢曢妶蹇曚覆闂傚倷绀侀幖顐ょ矓閺屻儱绀夐幖娣妼閻撴鈧箍鍎遍幊澶愬绩娴犲鐓熸俊顖氭惈缁狙呯磼娴ｅ弶娅婇柡灞稿墲閹峰懘宕妷鎰剁到鑿愰柛銉戝秷鍚梺璇″枟閻燂箓鏁嶉幇顑芥斀闁糕剝锕╅崬鍙夌節閻㈤潧浠滈柣掳鍔庨崚鎺斾沪閻愵剚鐝锋繛瀵稿Т椤戝懘鎮為崹顐犱簻闁圭儤鍨甸弸銈嗙箾閸忕厧濮堢紒缁樼洴瀹曪絾寰勭仦瑙ｅ悅闂佺粯鎸堕崐婵嬪蓟閿濆憘鐔封枎閹勵唲缂備胶鍋撻崕鎶解€﹂悜钘夎摕闁绘梻鍘х粈鍫㈡喐韫囨洘鏆滄繛鎴欏灪閻撶喖鏌熼幆褍鑸归柛鏃撶畵閺岀喖顢欑粵瀣杹闂佽桨鐒﹂崝鏍箚閺傚簱鏀介柛顐ゅ枑濞咃妇绱撻崒姘偓鐑芥倿閿曞倸绀夋俊銈呮噹绾惧鏌曢崼婵愭Ц闁藉啰鍠栭弻宥夊煛娴ｅ憡娈紓浣稿閸嬫盯鍩為幋锔藉亹闁圭粯甯╂禒濂告⒑閸愭彃妲婚柣妤佹尭椤繘鎼归崷顓犵厯濠电偛妫欓崕鎶藉礈閺夋娓婚柕鍫濈箻濡绢喖鈹戦悙璇ч練闁? {exc}")
            if "timeout" in str(exc).lower() or "connection" in str(exc).lower():
                logger.error("Network timeout detected; please verify local model cache or HF_ENDPOINT")

    def _should_skip_file(self, file_path: str, data_dir: str) -> bool:
        normalized = os.path.abspath(file_path)
        persist_dir = os.path.abspath(self.cfg["persist_dir"])
        if normalized.startswith(persist_dir):
            return True

        relative = os.path.relpath(normalized, os.path.abspath(data_dir)).replace("\\", "/")
        skip_prefixes = (
            "rag_evals/",
            "eval_results/",
        )
        excluded_files = {
            str(name).strip()
            for name in self.cfg.get("excluded_files", [])
            if str(name).strip()
        }
        skip_names = {
            "kb_chunk_preview.md",
            "rag_testcases.jsonl",
            "rag_testcases_enriched.jsonl",
            "intent_testcases.jsonl",
        }
        file_name = os.path.basename(relative)
        if relative.startswith(skip_prefixes):
            return True
        if file_name in skip_names or file_name in excluded_files:
            return True
        return False

    def _infer_subject(self, root: str, data_dir: str, file_name: str) -> str:
        root_name = os.path.basename(os.path.abspath(root))
        if root_name and root_name not in {"data", "documents"}:
            return root_name

        explicit_map = {
            "RAG\u6280\u672f\u539f\u7406\u7b80\u8ff0.md": "RAG",
            "README.md": "\u901a\u7528",
            "\u5982\u4f55\u9ad8\u6548\u5229\u7528AI\u8fdb\u884c\u5b66\u4e60.md": "\u5b66\u4e60\u65b9\u6cd5",
            "\u64cd\u4f5c\u7cfb\u7edf.pdf": "\u64cd\u4f5c\u7cfb\u7edf",
            "\u79bb\u6563\u6570\u5b66\u6838\u5fc3\u6982\u5ff5\u603b\u7ed3.md": "\u79bb\u6563\u6570\u5b66",
            "\u7a0b\u5e8f\u5458\u6c42\u804c\u9762\u8bd5\u4e0e\u7b80\u5386\u6307\u5357.md": "\u6c42\u804c\u9762\u8bd5",
            "\u8ba1\u7b97\u673a\u4e13\u4e1a\u6838\u5fc3\u8bfe\u7a0b\u5b66\u4e60\u6307\u5357.md": "\u5b66\u4e60\u65b9\u6cd5",
            "\u8ba1\u7b97\u673a\u4e13\u4e1a\u8bfe\u6838\u5fc3\u8003\u70b9\u6458\u8981.md": "\u901a\u7528",
            "\u8ba1\u7b97\u673a\u7ec4\u6210\u539f\u7406\u91cd\u70b9\u7b14\u8bb0.md": "\u8ba1\u7b97\u673a\u7ec4\u6210\u539f\u7406",
            "\u8f6f\u4ef6\u5de5\u7a0b\u4e0e\u5e38\u89c1\u8bbe\u8ba1\u6a21\u5f0f.md": "\u8f6f\u4ef6\u5de5\u7a0b",
        }
        if file_name in explicit_map:
            return explicit_map[file_name]

        lowered = file_name.lower()
        keyword_map = [
            (("\u79bb\u6563\u6570\u5b66",), "\u79bb\u6563\u6570\u5b66"),
            (("\u64cd\u4f5c\u7cfb\u7edf",), "\u64cd\u4f5c\u7cfb\u7edf"),
            (("\u7ec4\u6210\u539f\u7406", "\u8ba1\u7b97\u673a\u7ec4\u6210", "\u6307\u4ee4\u6267\u884c"), "\u8ba1\u7b97\u673a\u7ec4\u6210\u539f\u7406"),
            (("rag",), "RAG"),
            (("\u8f6f\u4ef6\u5de5\u7a0b", "\u8bbe\u8ba1\u6a21\u5f0f", "\u8f6f\u4ef6\u6d4b\u8bd5", "\u8c03\u8bd5"), "\u8f6f\u4ef6\u5de5\u7a0b"),
            (("\u6c42\u804c", "\u7b80\u5386", "\u9762\u8bd5"), "\u6c42\u804c\u9762\u8bd5"),
            (("\u5b66\u4e60\u6307\u5357", "\u5b66\u4e60\u65b9\u6cd5", "\u9ad8\u6548\u5229\u7528ai", "ai\u8fdb\u884c\u5b66\u4e60"), "\u5b66\u4e60\u65b9\u6cd5"),
        ]
        for keywords, subject in keyword_map:
            if any(keyword.lower() in lowered for keyword in keywords):
                return subject
        return "\u901a\u7528"

    def _infer_doc_role(self, file_name: str) -> str:
        if not file_name:
            return "normal"

        primary_files = {
            "RAG\u6280\u672f\u539f\u7406\u7b80\u8ff0.md",
            "README.md",
            "\u5982\u4f55\u9ad8\u6548\u5229\u7528AI\u8fdb\u884c\u5b66\u4e60.md",
            "\u64cd\u4f5c\u7cfb\u7edf\u5185\u5b58\u7ba1\u7406\u4e0e\u6587\u4ef6\u7cfb\u7edf\u4e13\u9898.md",
            "\u64cd\u4f5c\u7cfb\u7edf\u8fdb\u7a0b\u7ebf\u7a0b\u4e0e\u8c03\u5ea6\u4e13\u9898.md",
            "\u79bb\u6563\u6570\u5b66\u6838\u5fc3\u6982\u5ff5\u603b\u7ed3.md",
            "\u7a0b\u5e8f\u5458\u6c42\u804c\u9762\u8bd5\u4e0e\u7b80\u5386\u6307\u5357.md",
            "\u8ba1\u7b97\u673a\u4e13\u4e1a\u6838\u5fc3\u8bfe\u7a0b\u5b66\u4e60\u6307\u5357.md",
            "\u8ba1\u7b97\u673a\u4e13\u4e1a\u8bfe\u6838\u5fc3\u8003\u70b9\u6458\u8981.md",
            "\u8ba1\u7b97\u673a\u7ec4\u6210\u539f\u7406\u91cd\u70b9\u7b14\u8bb0.md",
            "\u8f6f\u4ef6\u5de5\u7a0b\u4e0e\u5e38\u89c1\u8bbe\u8ba1\u6a21\u5f0f.md",
        }
        if file_name in primary_files:
            return "primary"

        helper_files = {"urls.txt", "kb_chunk_preview.md"}
        if file_name in helper_files:
            return "helper"

        if file_name.lower().endswith(".pdf"):
            return "supplementary"

        priority_keywords = (
            "\u4e13\u9898",
            "\u6838\u5fc3\u77e5\u8bc6",
            "\u91cd\u70b9",
            "\u8be6\u89e3",
            "\u603b\u7ed3",
        )
        if file_name.lower().endswith(".md") and any(keyword in file_name for keyword in priority_keywords):
            return "primary"

        supplementary_keywords = (
            "\u5165\u95e8",
            "\u4e60\u9898",
            "\u8bef\u533a",
            "\u5bfc\u8bba",
            "\u57fa\u7840",
        )
        if any(keyword in file_name for keyword in supplementary_keywords):
            return "supplementary"

        return "normal"

    def _looks_like_noisy_text(self, text: str) -> bool:
        value = str(text or "")
        if not value.strip():
            return True
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", value):
            return True
        split_cjk = len(re.findall(r"[\u4e00-\u9fff]\s+[\u4e00-\u9fff]", value))
        if split_cjk >= 6:
            return True
        return False

    def _estimate_text_quality(self, text: str) -> float:
        value = str(text or "").strip()
        if not value:
            return -10.0

        score = 0.0
        compact = re.sub(r"\s+", "", value)
        if len(compact) >= 24:
            score += 1.0
        if len(compact) >= 80:
            score += 1.0
        if re.search(r"^#{1,3}\s+|^\s*[-*]\s+", value, flags=re.MULTILINE):
            score += 1.5
        if re.search(r"[.!?\u3002\uff01\uff1f]", value):
            score += 0.5
        if self._looks_like_noisy_text(value):
            score -= 6.0
        split_cjk = len(re.findall(r"[\u4e00-\u9fff]\s+[\u4e00-\u9fff]", value))
        score -= min(split_cjk, 10) * 0.5
        return score

    def _score_doc_source_bias(self, doc: Any, query_terms: List[str]) -> float:
        metadata = getattr(doc, "metadata", {}) or {}
        content = str(getattr(doc, "page_content", "") or "")
        file_name = str(metadata.get("file_name") or "").strip().lower()
        chunk_title = str(metadata.get("chunk_title") or metadata.get("title") or "").strip().lower()
        document_title = str(metadata.get("document_title") or "").strip().lower()
        title_blob = " ".join(part for part in [file_name, chunk_title, document_title] if part)
        quality = self._estimate_text_quality(content)
        pdf_bias = float(self.cfg.get("pdf_boost", 0.15))

        score = 0.0
        if file_name.endswith(".md") or metadata.get("source") == "markdown_file":
            score += 0.12
        elif file_name.endswith(".txt"):
            score += 0.06
        elif file_name.endswith(".pdf") or metadata.get("source") == "pdf_file":
            score -= min(0.22, max(0.08, pdf_bias * 0.35))
            if quality >= 2.5:
                score += min(0.06, pdf_bias * 0.1)

        if quality < 0:
            score -= min(0.28, abs(quality) * 0.03)
        if self._looks_like_noisy_text(content) or self._is_low_quality_text(content):
            score -= 0.18

        for term in query_terms:
            if not term:
                continue
            if term == chunk_title:
                score += 0.22
            elif term in chunk_title or term in document_title:
                score += 0.14
            elif term in title_blob:
                score += 0.1
        return score

    def _tokenize_for_diversity(self, text: str) -> List[str]:
        parts = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{2,}", str(text or "").lower())
        tokens: List[str] = []
        for part in parts:
            if re.fullmatch(r"[\u4e00-\u9fff]{2,}", part):
                if len(part) == 2:
                    tokens.append(part)
                else:
                    tokens.extend(part[idx:idx + 2] for idx in range(len(part) - 1))
            else:
                tokens.append(part)
        return tokens

    def _doc_subject_for_ranking(self, doc: Any) -> str:
        metadata = getattr(doc, "metadata", {}) or {}
        subject = str(metadata.get("subject") or "").strip()
        if subject and subject != "通用":
            return subject

        file_name = str(metadata.get("file_name") or "").strip()
        file_path = str(metadata.get("file_path") or "").strip()
        if file_name:
            return self._infer_subject("", self._documents_dir, file_name)
        if file_path:
            root = os.path.dirname(file_path)
            return self._infer_subject(root, self._documents_dir, os.path.basename(file_path))
        return "通用"

    def _doc_source_key(self, doc: Any) -> str:
        metadata = getattr(doc, "metadata", {}) or {}
        return str(
            metadata.get("file_name")
            or metadata.get("file_path")
            or metadata.get("source")
            or ""
        ).strip().lower()

    def _doc_title_key(self, doc: Any) -> str:
        metadata = getattr(doc, "metadata", {}) or {}
        title = str(
            metadata.get("chunk_title")
            or metadata.get("title")
            or metadata.get("document_title")
            or ""
        ).strip().lower()
        if title:
            return title
        content = str(getattr(doc, "page_content", "") or "").strip().lower()
        return content[:80]

    def _doc_similarity(self, left: Any, right: Any) -> float:
        left_meta = getattr(left, "metadata", {}) or {}
        right_meta = getattr(right, "metadata", {}) or {}

        left_title = " ".join(
            str(left_meta.get(key) or "").strip().lower()
            for key in ("chunk_title", "title", "document_title")
        )
        right_title = " ".join(
            str(right_meta.get(key) or "").strip().lower()
            for key in ("chunk_title", "title", "document_title")
        )
        left_text = f"{left_title} {(str(getattr(left, 'page_content', '') or '')[:220]).lower()}"
        right_text = f"{right_title} {(str(getattr(right, 'page_content', '') or '')[:220]).lower()}"

        left_tokens = set(self._tokenize_for_diversity(left_text))
        right_tokens = set(self._tokenize_for_diversity(right_text))
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    def _select_diverse_documents(
        self,
        ranked_docs: List[Any],
        top_k: int,
    ) -> List[Any]:
        if len(ranked_docs) <= top_k:
            return ranked_docs[:top_k]

        diversity_threshold = float(self.cfg.get("diversity_threshold", 0.3))
        max_chunks_per_source = max(1, int(self.cfg.get("max_chunks_per_source", 4)))
        max_chunks_per_title = max(1, int(self.cfg.get("max_chunks_per_title", 1)))

        selected: List[Any] = []
        selected_ids = set()
        source_counts: Counter[str] = Counter()
        title_counts: Counter[str] = Counter()

        def try_add(doc: Any, enforce_similarity: bool, source_limit: int, title_limit: int) -> bool:
            doc_id = id(doc)
            if doc_id in selected_ids:
                return False

            source_key = self._doc_source_key(doc)
            title_key = self._doc_title_key(doc)
            if source_key and source_counts[source_key] >= source_limit:
                return False
            if title_key and title_counts[title_key] >= title_limit:
                return False

            if enforce_similarity:
                for existing in selected:
                    if self._doc_similarity(doc, existing) >= diversity_threshold:
                        return False

            selected.append(doc)
            selected_ids.add(doc_id)
            if source_key:
                source_counts[source_key] += 1
            if title_key:
                title_counts[title_key] += 1
            return True

        for doc in ranked_docs:
            if len(selected) >= top_k:
                break
            try_add(doc, enforce_similarity=True, source_limit=max_chunks_per_source, title_limit=max_chunks_per_title)

        if len(selected) < top_k:
            for doc in ranked_docs:
                if len(selected) >= top_k:
                    break
                try_add(
                    doc,
                    enforce_similarity=False,
                    source_limit=max_chunks_per_source + 1,
                    title_limit=max_chunks_per_title + 1,
                )

        if len(selected) < top_k:
            for doc in ranked_docs:
                if len(selected) >= top_k:
                    break
                doc_id = id(doc)
                if doc_id in selected_ids:
                    continue
                selected.append(doc)
                selected_ids.add(doc_id)

        return selected[:top_k]

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""

        # Remove CJK radicals / compatibility glyphs that often appear in corrupted PDF extraction.
        text = re.sub(r"[\u2E80-\u2FDF\uF900-\uFAFF]", " ", text)

        lines = text.splitlines()
        cleaned_lines: List[str] = []
        seen_lines = set()

        for raw_line in lines:
            line = raw_line.strip()
            if not line or len(line) < 2:
                continue
            if line in seen_lines:
                continue
            if re.match(r"^(\u7b2c?\s*\d+\s*\u9875|\d+/\d+|\d+)$", line):
                continue
            if re.match(r"^=+\s*\u7b2c?\s*\d+\s*\u9875\s*=+$", line):
                continue
            if re.search(r"([.\uFF0E\u00B7\u2022\u2026]{4,}|-{5,}|_{5,})", line):
                continue
            if re.match(r"^[^\n]{1,80}\s\d{1,3}$", line):
                continue
            line = re.sub(r"[ \t\u3000]+", " ", line)
            seen_lines.add(line)
            cleaned_lines.append(line)

        text = "\n".join(cleaned_lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_heading_title(self, line: str) -> str:
        stripped = str(line or "").strip()
        if not stripped:
            return ""

        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", stripped)
        if not match:
            return ""

        title = re.sub(r"[`*_]+", "", match.group(2)).strip(" -:?#")
        if len(title) < 2 or len(title) > 40:
            return ""
        return title

    def _split_document_by_headings(self, doc: Any) -> List[Any]:
        if self.imports is None:
            return [doc]

        Document = self.imports["Document"]
        content = getattr(doc, "page_content", "") or ""
        metadata = dict(getattr(doc, "metadata", {}) or {})
        lines = content.splitlines()

        sections: List[Any] = []
        current_lines: List[str] = []
        current_title = str(metadata.get("chunk_title") or "")
        document_title = str(metadata.get("document_title") or "")
        heading_count = 0

        def flush_section() -> None:
            nonlocal current_lines
            if not current_lines:
                return
            section_text = "\n".join(current_lines).strip()
            if not section_text:
                current_lines = []
                return
            section_meta = dict(metadata)
            if document_title:
                section_meta["document_title"] = document_title
            if current_title:
                section_meta["chunk_title"] = current_title
            sections.append(Document(page_content=section_text, metadata=section_meta))
            current_lines = []

        for raw_line in lines:
            heading = self._extract_heading_title(raw_line)
            if heading:
                heading_count += 1
                if not document_title:
                    document_title = heading
                if current_lines:
                    flush_section()
                current_title = heading
                current_lines = [raw_line.strip()]
                continue
            current_lines.append(raw_line)

        flush_section()
        return sections if heading_count > 0 and sections else [doc]

    def _prefix_chunk_title(self, content: str, metadata: Dict[str, Any]) -> str:
        text = str(content or "").strip()
        if not text:
            return ""

        chunk_title = str(metadata.get("chunk_title") or "").strip()
        if not chunk_title:
            return text
        if chunk_title in text[:120]:
            return text
        return f"## {chunk_title}\n{text}"

    def _is_low_quality_text(self, text: str) -> bool:
        if not text:
            return True

        compact = re.sub(r"\s+", "", text)
        if len(compact) < 20:
            return True

        meaningful = re.findall(r"[\u4e00-\u9fffA-Za-z]", compact)
        meaningful_ratio = len(meaningful) / max(len(compact), 1)
        dotted_ratio = len(re.findall(r"[.\uFF0E\u00B7\u2022\u2026]", compact)) / max(len(compact), 1)
        radical_ratio = len(re.findall(r"[\u2E80-\u2FDF\uF900-\uFAFF]", compact)) / max(len(compact), 1)
        digit_ratio = len(re.findall(r"\d", compact)) / max(len(compact), 1)
        repeated_symbol_ratio = len(re.findall(r"[=\-_]{3,}", compact)) / max(len(compact), 1)
        return (
            meaningful_ratio < 0.35
            or dotted_ratio > 0.20
            or radical_ratio > 0.08
            or digit_ratio > 0.35
            or repeated_symbol_ratio > 0.10
        )

    def _split_documents(self, docs: List[Any]) -> List[Any]:
        if not docs or self.imports is None:
            return []

        splitter = self.imports["RecursiveCharacterTextSplitter"](
            chunk_size=self.cfg["chunk_size"],
            chunk_overlap=self.cfg["chunk_overlap"],
            separators=["\n\n", "\n", "\u3002", "\uff01", "\uff1f", "\uff1b", " ", ""],
        )

        section_docs: List[Any] = []
        for doc in docs:
            section_docs.extend(self._split_document_by_headings(doc))

        split_docs = splitter.split_documents(section_docs)
        filtered_docs = []
        for doc in split_docs:
            metadata = dict(getattr(doc, "metadata", {}) or {})
            content = self._clean_text(getattr(doc, "page_content", "") or "")
            content = self._prefix_chunk_title(content, metadata)
            if not content or self._is_low_quality_text(content):
                continue
            doc.metadata = metadata
            doc.page_content = content
            filtered_docs.append(doc)
        return filtered_docs

    def _collect_documents(self) -> List[Any]:
        if self.imports is None:
            return []

        try:
            Document = self.imports["Document"]
            os.makedirs(self._data_dir, exist_ok=True)
            os.makedirs(self._documents_dir, exist_ok=True)
            documents: List[Any] = []

            def add_document(text: str, metadata: Dict[str, Any]) -> None:
                cleaned = self._clean_text((text or "").strip())
                if not cleaned:
                    return
                is_pdf = metadata.get("source") == "pdf_file"
                min_length = 20 if is_pdf else int(self.cfg["min_content_length"])
                if len(cleaned) < min_length or self._is_low_quality_text(cleaned):
                    return
                documents.append(Document(page_content=cleaned, metadata=metadata))

            for root, _, files in os.walk(self._documents_dir):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    if self._should_skip_file(file_path, self._documents_dir):
                        continue

                    lower_name = file_name.lower()
                    try:
                        if lower_name.endswith(".txt"):
                            with open(file_path, "r", encoding="utf-8", errors="replace") as file:
                                add_document(
                                    file.read(),
                                    {
                                        "source": "file",
                                        "type": "text",
                                        "file_name": file_name,
                                        "file_path": file_path,
                                        "subject": self._infer_subject(root, self._documents_dir, file_name),
                                    },
                                )
                        elif lower_name.endswith(".json"):
                            with open(file_path, "r", encoding="utf-8", errors="replace") as file:
                                content = json.load(file)
                            if isinstance(content, dict) and "text" in content:
                                meta = content.get("meta", {}) if isinstance(content.get("meta"), dict) else {}
                                add_document(
                                    str(content["text"]),
                                    {
                                        "source": "file",
                                        "type": "json",
                                        "file_name": file_name,
                                        "file_path": file_path,
                                        "subject": self._infer_subject(root, self._documents_dir, file_name),
                                        **meta,
                                    },
                                )
                    except Exception as exc:
                        logger.error(f"濠电姷鏁告慨鐑藉极閸涘﹥鍙忛柣鎴ｆ閺嬩線鏌涘☉姗堟敾闁告瑥绻愰湁闁稿繐鍚嬬紞鎴︽煕閵娿儱鈧潡寮婚敐澶婄鐎规洖娲ら崫娲⒑閸濆嫷鍎愰柣妤侇殘閹广垹鈽夐姀鐘殿吅闂佺粯鍔樼亸娆撳礉娴煎瓨鈷戠紒瀣儥閸庢劙鏌熼崨濠冨€愰柨婵堝仜閳规垹鈧綆鍋勬禍妤呮煙閸忚偐鏆橀柛銊︽閵囨劖鎯旈～顓犵畾闂侀潧鐗嗗ú銈呮毄闂備胶顭堥鍡涙儎椤栨氨鏆︽い鏍剱閺佸秵绻濇繝鍌氼仼閹兼潙锕ら埞鎴︽偐缂佹ɑ閿銈嗗灥閹虫﹢骞冨▎鎰瘈闁告洦鍓欓弸鎴︽煙閸忚偐鏆橀柛鎿冨亝瀵板嫭绻涢悙顒佹澑闂佽鍑界紞鍡涘磻閳ь剟鏌熼柨瀣仢婵﹨娅ｅ☉鐢稿川椤撴繃鐫忛梻浣侯焾椤戝棝鎯勯姘辨殾婵犻潧妫銊╂煃瑜滈崜鐔奉嚕鐠囨祴妲堥柕蹇曞У椤ユ繈姊洪崨濠勫彋闁靛牆鎳忛崕鎾绘⒒閸屾瑨鍏岀痪顓炵埣瀵剚绗熼埀顒€鐣烽幋锕€绠婚柟棰佺劍鐎靛矂姊洪棃娑氬婵☆偅顨婂畷鍛婄節閸ャ劎鍘遍柣搴秵閸嬪懎鐣峰畝鍕厸閻忕偛澧介埊鏇犵磼缂佹绠炵€规洜鍠栭、妤呭焵椤掆偓閳绘挾浠︾粵瀣瘜闂侀潧鐗嗗Λ娆戝閹间焦鐓曢幖娣妽濞懷囨煟閿濆洤鍘撮柟顔哄灮閸犲﹥娼忛妸锔界彍婵犵數鍋犻幓顏嗙礊閳ь剚绻涢崪鍐К闁稿﹤鐡ㄧ换婵堝枈婢跺瞼锛熼梺绋款儐閸ㄥ灝鐣烽幇鏉垮唨妞ゆ挾鍋熼ˇ顕€姊虹涵鍛涧缂佺姵鍨规竟? {file_path}, 闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻锝夊箣閿濆憛鎾绘煕婵犲倹鍋ラ柡灞诲姂瀵噣宕奸悢鍛婎唶闂備胶顭堥鍡涘箰閸撗冨灊妞ゆ挾鍋愬Σ鍫熶繆椤栨繍鍤欐い锔规櫊濮婅櫣绮欏▎鎯у妧缂備胶绮换鍫濐嚕婵犳艾鍗抽柣鏃囨椤旀洟姊虹紒妯哄Е闁告挻宀搁幃鐢告晝閸屾稈鎷? {exc}")

            from rag.content_loader import clean_pdf_text, fetch_url, has_meaningful_content, load_markdown, load_pdf
            import io

            supported_exts = {
                ".pdf": "pdf_file",
                ".md": "markdown_file",
                ".markdown": "markdown_file",
                ".py": "code_file",
                ".js": "code_file",
                ".ts": "code_file",
                ".java": "code_file",
                ".c": "code_file",
                ".cpp": "code_file",
                ".html": "code_file",
                ".css": "code_file",
            }

            for root, _, files in os.walk(self._documents_dir):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    if self._should_skip_file(file_path, self._documents_dir):
                        continue

                    ext = os.path.splitext(file_name)[1].lower()
                    if ext not in supported_exts:
                        continue

                    try:
                        with open(file_path, "rb") as file:
                            stream = io.BytesIO(file.read())

                        if ext == ".pdf":
                            text = load_pdf(stream)
                            if text and not has_meaningful_content(text):
                                text = clean_pdf_text(text)
                        else:
                            text = load_markdown(stream)

                        add_document(
                            text,
                            {
                                "source": supported_exts[ext],
                                "subject": self._infer_subject(root, self._documents_dir, file_name),
                                "file_name": file_name,
                                "file_path": file_path,
                            },
                        )
                    except Exception as exc:
                        logger.error(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾剧懓顪冪€ｎ亝鎹ｉ柣顓炴閵嗘帒顫濋敐鍛婵°倗濮烽崑娑⑺囬悽绋垮瀭濡わ絽鍟粻娑樏归敐鍛础缂佸鑳剁槐鎾诲磼濮橆兘鍋撻悜鑺ュ€块柨鏇氱劍閹冲矂姊绘担鍛婂暈闁荤喆鍎抽幑銏狀潨閳ь剙顕ｆ繝姘亜缁炬媽椴搁弲顒勬⒑閹稿海绠撴繛璇х到鏁堟俊銈呮噺閸嬧剝绻濇繝鍌涘櫣妞わ絽銈搁幃浠嬵敍濞戞ɑ璇為梺璇″枟閻燂妇鎹㈠┑瀣闁挎棁娉曢惄搴ㄦ⒒娴ｇ儤鍤€闁圭⒈鍋勮灋婵°倓鑳堕々鏌ユ煙椤栧棌鍋撻柡鈧禒瀣厓闁芥ê顦伴ˉ婊兠瑰鍕疄闁哄被鍔戝鏉懳熼搹閫涚礄闁诲孩顔栭崰娑㈩敋瑜旈崺銉﹀緞閹邦剦娼婇梺缁樕戦悧鎾诲礉閹达箑钃熼柕濞垮劗閺€浠嬫煕閳╁厾顏勨枍閿濆鐓欐い鏍ㄧ矊閺嬫稒鎱ㄦ繝鍕笡缂佹鍠栭崺鈧い鎺戝閻掑灚銇勯幒鎴濃偓鍛婄濠婂牊鐓犳繛鑼额嚙閻忥繝鏌￠崨顓犲煟妞ゃ垺宀搁崺鈧い鎺戝閽冪喐绻涢幋鐐垫噮缂佲檧鍋撻梻浣告啞濞诧箓宕滃▎蹇婃灁閻忕偟鐡斿〒濠氭煏閸繃顥炵痪鐐倐閺屾稒鎯旈妸锔介敪闂佷紮绲块崗姗€骞冮姀銏犳瀳閺夊牄鍔嶅▍灞解攽閻愯埖褰х紓宥佸亾濠电偛鎷戠紞鈧柛濠傜摠缁绘繄鍠婃径宀€锛熼梺绋款儐閸ㄥ灝鐣烽幇鏉垮唨妞ゆ挾鍋熼ˇ顕€姊虹涵鍛涧缂佺姵鍨规竟? {file_path}, 闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻锝夊箣閿濆憛鎾绘煕婵犲倹鍋ラ柡灞诲姂瀵噣宕奸悢鍛婎唶闂備胶顭堥鍡涘箰閸撗冨灊妞ゆ挾鍋愬Σ鍫熶繆椤栨繍鍤欐い锔规櫊濮婅櫣绮欏▎鎯у妧缂備胶绮换鍫濐嚕婵犳艾鍗抽柣鏃囨椤旀洟姊虹紒妯哄Е闁告挻宀搁幃鐢告晝閸屾稈鎷? {exc}")

            try:
                if os.path.exists(self._url_sources_path):
                    with open(self._url_sources_path, "r", encoding="utf-8") as file:
                        url_items = json.load(file)
                    if isinstance(url_items, list):
                        for item in url_items:
                            if not isinstance(item, dict) or not item.get("url"):
                                continue
                            try:
                                text, meta = fetch_url(str(item["url"]))
                                add_document(
                                    text,
                                    {
                                        "source": "url",
                                        "type": "web",
                                        "file_name": meta.get("title") or str(item["url"]),
                                        "file_path": str(item["url"]),
                                        "subject": item.get("subject", "general"),
                                        "url": str(item["url"]),
                                    },
                                )
                            except Exception as exc:
                                logger.warning(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌ｉ幋锝呅撻柛銈呭閺屾盯顢曢敐鍡欙紭闂侀€炲苯鍘搁柣鎺炲閹广垹鈹戠€ｎ亞鍊為悷婊冮鍗辨い鎺戝閳锋垿鏌涘┑鍡楊伀闁宠顦甸弻娑樜熼崹顔绘睏缂備緡鍠栭…鐑藉箖瑜斿畷濂告偄瀹勯绱熷┑鐘垫暩婵炩偓婵炰匠鍏炬稑顭ㄩ崼顒傜◤闂佹寧娲嶉崑鎾绘?URL 闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌ｉ幋锝呅撻柛濠傛健閺屻劑寮村Δ鈧禍鎯ь渻閵堝簼绨婚柛鐔告綑閻ｉ绮欑拠鐐閸┾偓妞ゆ帒瀚壕濂告煏婵炑冨閺嬫垿姊虹紒姗嗘當闁绘妫涚划顓㈠箳閺冨倹锛忛梺璇″瀻閸曨偂娣梻浣告惈閻ジ宕伴幘鑸殿潟闁圭儤顨呴～鍛存煟濡櫣锛嶅ù婊冪埣濮婄粯鎷呮笟顖滃姼婵°倗濮甸幃鍌炲极閸愵噮鏁傞柛顐ｇ箚閹芥洟姊虹紒妯活梿闁靛棌鍋撻梺绋款儐閹瑰洭骞冮挊澶嗘灁閻犳亽鍔屾慨鐓庘攽閻愬樊鍤熷┑顔炬暩閸掓帗鎯旈敐鍡╂綗闂佸湱鍎ら〃鍛矆閸緷褰掓晲婢跺鐝冲┑鈥冲级閸旀瑩骞冨Δ鈧埢鎾诲垂椤旂晫浜梻浣虹帛閻楁粓宕㈣閹箖鏌ㄧ€ｎ剟妾梺鍛婃尭瀵爼寮插┑鍡╂富闁靛牆楠告禍浠嬫煕濡亽鍋㈢€规洜鏁婚弻鍡楊吋閸℃瑥骞? {item.get('url')}, 闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻锝夊箣閿濆憛鎾绘煕婵犲倹鍋ラ柡灞诲姂瀵噣宕奸悢鍛婎唶闂備胶顭堥鍡涘箰閸撗冨灊妞ゆ挾鍋愬Σ鍫熶繆椤栨繍鍤欐い锔规櫊濮婅櫣绮欏▎鎯у妧缂備胶绮换鍫濐嚕婵犳艾鍗抽柣鏃囨椤旀洟姊虹紒妯哄Е闁告挻宀搁幃鐢告晝閸屾稈鎷? {exc}")
            except Exception as exc:
                logger.warning(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾剧懓顪冪€ｎ亝鎹ｉ柣顓炴閵嗘帒顫濋敐鍛婵°倗濮烽崑娑⑺囬悽绋垮瀭濡わ絽鍟粻娑樏归敐鍛础缂佸鑳剁槐鎾诲磼濮橆兘鍋撻悜鑺ュ€块柨鏇氱劍閹冲矂姊绘担鍛婂暈闁荤喆鍎抽幑銏狀潨閳ь剙顕ｆ繝姘亜缁炬媽椴搁弲顒勬⒑閹稿海绠撴繛璇х到鏁?URL 闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌ｉ幋锝呅撻柛濠傛健閺屻劑寮村Δ鈧禍鎯ь渻閵堝簼绨婚柛鐔告綑閻ｉ绮欑拠鐐閸┾偓妞ゆ帒瀚壕濂告煏婵炑冨閺嬫垿姊虹紒姗嗘當闁绘妫涚划顓㈠箳閺冨倹锛忛梺璇″瀻閸曨偂娣梻浣告惈閻ジ宕伴幘鑸殿潟闁圭儤顨呴～鍛存煟濡櫣锛嶅ù婊冪埣濮婄粯鎷呮笟顖滃姼婵°倗濮甸幃鍌炲极閸愵噮鏁傞柛顐ｇ箚閹芥洟姊虹紒妯活梿闁靛棌鍋撻梺绋款儐閹告悂鍩ユ径濞炬瀻婵☆垳鍘ф慨鍝勨攽閻樻鏆柍褜鍓濆▍鏇㈠窗濮椻偓閺屸€崇暆鐎ｎ剛鐦堥梺绯曟杹閸嬫挸顪冮妶鍡楃瑨閻庢凹鍙冨鎻掆攽鐎ｎ偆鍘遍柣蹇曞仜婢т粙濡撮幒妤佺厱婵☆垰鍚嬮弳顒佹叏婵犲嫮甯涚紒妤冨枛瀹曟儼顦插ù鐓庢捣缁辨挻鎷呴崫鍕戯綁鏌涢妸銉т虎闁伙絿鍏樻慨鈧柕鍫濇－濡啫鈹戦悙瀵告殬闁搞劌顭烽獮濠囧箻閻戔斁鍋撻敃鍌毼╅柍杞拌兌椤︺劑姊洪幐搴㈢５闁稿鎸搁湁闁绘瑢鍋撻柛锝忕到椤? {exc}")

            return documents
        except Exception as exc:
            logger.error(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧湱鈧懓瀚崳纾嬨亹閹烘垹鍊炲銈嗗坊閸嬫挻娼诲┑瀣拺闂侇偆鍋涢懟顖涙櫠閹绢喗鐓涢悘鐐插⒔閵嗘帡鏌嶈閸撱劎绱為崱娑樼；闁告侗鍘鹃弳锔锯偓鍏夊亾闁告洦鍓涢崢鎼佹煟韫囨洖浠滃褑妫勭叅閻庢稒菧娴滄粓鏌曡箛濠傚⒉缂佲偓鐎ｎ喗鐓涘ù锝囨嚀婵秶鈧娲栧畷顒勫煝鎼粹垾鐔煎箒瀹ュ洤浠辨慨濠傤煼瀹曟帒鈻庨幇顔哄仒濠电姰鍨婚幊鎾绘偋韫囨梻顩茬紒瀣氨閺嬪酣鏌熼柇锕€鏋ゆい鏃€娲熷娲焻閻愯尪瀚板褎鎸抽弻锛勪沪缁洖浜剧€规洖娲﹀▓鏇㈡煟鎼搭垳绉甸柛鎾寸洴閹線宕奸妷锕€鈧敻鎮峰▎蹇擃仾缂佲偓閸愵喗鍋ㄦい鏍ㄧ☉濞搭噣鏌ㄥ┑鍫濅粶闁宠鍨归埀顒婄到閻忔岸寮查敐澶嬧拺缂備焦锚閻忋儲淇婇锝囨噯闁愁亝鎮傚濠氬磼濮橆兘鍋撻幖浣哥９濡炲瀛╅鑺ユ叏濮楀棗澧婚柣鎺旀櫕閹茬顓奸崼顫綍濠电姴鐥夐弶搴撳亾閺囥垹绠犻幖杈剧到椤ユ氨鎲搁悧鍫濈瑲闁绘挾鍠栭弻锝夊籍閸屾艾浠樻繛瀛樼矆閸楁娊寮婚敐澶婄闁哄洨鍣ュΛ鍕⒑鐎圭媭娼愰柛銊ユ健楠炲啴鍩￠崘锝呬壕闁革富鍘煎瓭濡炪們鍎茬换鍫濐潖閾忓湱纾兼俊顖氭惈椤矂姊洪幐搴㈠濞存粠浜滈锝囨嫚瀹割喖鎮戞繝銏ｆ硾閿曘儱鐣靛鍜佹富闁靛牆妫欑亸鐢告煕鐎ｎ倖鎴犵博閻旂厧鍗抽柕蹇婃閹疯櫣绱撻崒娆戝妽閽冮亶鎮樿箛娑欐锭闁宠鍨块、娆撴倷椤掍焦鐦庨梻浣烘嚀缁犲秹宕硅ぐ鎺戠叀濠㈣埖鍔曢～鍛存煟濡椿鍟忛柛姘嚇濮婄粯鎷呴搹鐟扮闂佸憡姊瑰ú鐔煎箖濞差亜惟闁宠桨鑳堕敍鐔兼煙閸忚偐鏆橀柛鏂跨Ч閸╂盯骞嬮敂鐣屽幈濠电偞鍨堕敃鈺呮偂椤掑嫭鐓熼柟鐑樻尭閸斻倖銇勯鐐村仴闁诡喚鍏橀獮宥夘敊鐏忔牕寰嶆繝鐢靛Х椤ｎ喚妲愰弴銏犳瀬濠电姵姘ㄥ畵渚€鏌涢妷顔煎缂佺嫏鍥ㄥ仯濞撴凹鍨抽崢婊呯磼鏉堛劎鍙€婵﹦绮幏鍛存惞閻熸壆顐奸梻浣告啞濮婂綊鎮ч弴鈶┾偓锕傚炊瑜夐弸搴ㄦ煙闁箑娅樻繛鏉戝閳规垿鎮欓崣澶嗘灆婵炲瓨绮嶇换鍫ュ春濞戙垹绠ｉ柣妯兼暩閿涙繃绻涙潏鍓хК闁稿鍊块獮瀣晜閽樺绨甸梻渚€娼чˇ顐﹀疾濠婂牊鍋傞柡鍥╁枑閸欏繑鎱ㄥΔ鈧Λ妤佹櫠婵犳碍鐓曢幖娣灪閳锋帡鏌嶈閸撴岸顢欓弽顓炵獥婵°倕鎳忛崑鍌炴煥閻斿搫孝闁哄嫨鍎甸弻銊╂偄閸濆嫅銏⑩偓鐟版啞缁诲啴濡甸崟顖氱睄闁割偆鍣︾槐鐢告⒑? {exc}", exc_info=True)
            return []

    def _extract_docs_from_loaded_store(self) -> List[Any]:
        if self.vs is None:
            return []

        docs: List[Any] = []
        try:
            docstore = getattr(self.vs, "docstore", None)
            index_to_docstore_id = getattr(self.vs, "index_to_docstore_id", None)
            if not docstore or not index_to_docstore_id:
                return []

            for docstore_id in index_to_docstore_id.values():
                try:
                    doc = docstore.search(docstore_id)
                    if doc is not None:
                        docs.append(doc)
                except Exception:
                    continue
        except Exception as exc:
            logger.warning(f"濠电姷鏁告慨鐑藉极閸涘﹥鍙忛柣鎴ｆ閺嬩線鏌熼梻瀵割槮缁惧墽绮换娑㈠箣濞嗗繒鍔撮梺杞扮椤戝棝濡甸崟顖氱閻犺櫣鍎ら悗楣冩⒑閸涘﹦鎳冪紒缁橈耿楠炲啫螖閳ь剟鍩ユ径濞炬瀻婵炲棗娴锋潻鏃€绻濋悽闈涗粶闁绘鎳愰崚鎺戔枎閹惧疇鎽曢梺缁樻煥閸氬鍩涢幒妤佸€甸梻鍫熺☉椤ュ繑淇婂顔兼灓缂佽鲸鎸婚幏鍛村箵閹哄秴顥氶梻鍌欑閹测€趁洪敃鍌氬瀭闂侇剙绉甸崑鍌炴煏婢跺棙娅嗛柣鎾跺枛閺岋繝宕掑☉姗嗗殝闂佽鍨伴悧蹇涘焵椤掍胶鈯曠紒璇插濡叉劙骞樼€涙ê顎撻梺鍦帛鐢﹥绔熼弴鐐╂斀闁绘﹩鍠栭悘閬嶆煕閳哄倻澧电€规洘绻堝浠嬵敇閻旇渹绨甸梻浣虹帛濡啴寮查懠顒佸床闁糕剝菧娴滄粓鏌″鍐ㄥ闁活厼锕幃浠嬵敍濮樺崬鐓熼梺鍝勬湰閻╊垰顕ｉ鈧埢搴ㄥ箚瑜嶆竟瀣磽閸屾瑦绁板鏉戞憸閺侇喖螖閸涱厾顔嗛梺鍛婁緱閸橀箖寮崇€ｎ喗鐓涘璺哄绾爼鏌曢崼顐㈡珝婵﹦绮幏鍛村川婵犲啫鍓甸梻浣告惈閻楁粓宕滃☉姘灊婵炲棙鎸哥粻锝夋煥閺冨洤浜圭紒瀣箰閳规垿鎮╃紒妯婚敪濡炪倖鍨堕崝娆忣嚕椤愶箑鐐婃い鎺嶈兌閸樹粙姊鸿ぐ鎺戜喊闁告挻鐟╅獮鍐╂媴閾忓湱锛滈柡澶婄墑閸斿海绮婚悙瀛樺弿濠电姴瀚敮娑氱磼濡ゅ啫鏋涢柛鈹惧亾濡炪倖宸婚崑鎾淬亜椤撶偞鍠橀柟绛圭節婵″爼宕ㄩ婊呮憣闂傚倷鑳堕～瀣礋閸偆鏆ョ紓浣哄亾閸庡啿顭囬敓鐘茶摕闁哄洢鍨归柋鍥ㄧ節閸偄濮堢粭鎴犵磽閸屾瑨鍏屽┑顕€娼ч悾婵嬪箹娴ｆ瓕鎽曢梺璺ㄥ枔婵绮堥崘顔界厪濠电倯鍐ㄦ灓闁哄棭鍋嗙槐鎾诲磼濞嗘帒鍘℃繝娈垮枟閸庢娊鎮鹃悜鑺ュ仺闁割煈鍋掑鐔兼⒑鐟欏嫬绀冩い鏇嗗洦鐓侀柛銉墯閻撳繘鐓崶銊︾妞ゎ偅鍨块弻鏇＄疀鐎ｎ亖鍋撻弽顓熷亗闁哄洢鍨洪悡蹇撯攽閻樿尙绠版い鈺婂墰缁辨帡鎮╁畷鍥р拡缂備浇椴哥敮鐐垫閹烘嚦鐔兼嚃閳哄啯婢栨繝鐢靛У椤旀牠宕伴幒妤€纾婚柟鍓х帛閳锋垿鏌熺粙鎸庢崳缂佺姵鎸绘穱濠囶敃閿濆洦鍒涙繝纰樺墲閹告娊鐛崶顒夋晩闁告挆宥囬棷婵犵數鍋犻幓顏嗗緤閼恒儳鐝跺┑鐘蹭迹閿濆悿娲敂閸涱垰骞愬┑鐐舵彧缁插潡鎮洪弮鍫濆惞婵炲棙鍔戞禍婊堟煛閸ユ湹绨界紒澶樺枟椤ㄣ儵鎮欑€电顫ч梺鐟板槻閹虫ê鐣烽锕€唯妞ゆ柨褰ㄩ悙娴嬫斀闁挎稑瀚禍濂告煕婵炲灝鈧繂鐣烽姀锛勵浄閻庯綆浜滈悗顓㈡⒑閸涘﹥灏柤鍐茬埣閹本寰勯幇顓犲幈闂佹枼鏅涢崯浼村煀閺囥垺鐓冮梺鍨儏閻忔挳鏌″畝鈧崰鎾跺垝濞嗘挸绠伴幖娣灩闂傤垳绱撻崒娆掑厡濠殿喚鏁婚獮鎰版嚒閵堝棗搴婂┑鐐村灦閻喖鈻介鍫熺厱閻忕偞鍎抽悞褰掓煕鐎ｎ偅灏柍瑙勫灴瀹曞崬鈻庨幇顓у晭闂傚倷鑳堕幊鎾绘偤閵娧勫床鐎广儱顦憴锕傛倵閿濆骸鏋熼柣鎾存礋閺屾洘寰勫Ο铏逛化闂佸憡甯楀姗€婀佸┑鐘诧工閹冲孩绂嶉崜褏纾肩紓浣诡焽濞插鈧娲滈崢褔鍩為幋锕€閱囨繝濠傛椤庢姊婚崒娆戭槮闁圭⒈鍋婇、鏍炊椤掆偓缁愭鏌熼柇锕€鏋熺€规洖寮剁换娑㈠箣濞嗗繒浠奸梺娲诲幗椤ㄥ懘鈥︾捄銊﹀磯濡わ箑鐏濋顓㈡⒑閸愭彃妲婚柡鍫墰閹广垹鈹戦崼婵囩€虫繝銏ｆ硾閻ジ鏁嶉悢鍏尖拺閺夌偞澹嗛ˇ锔姐亜椤撶姴鍘撮柟顔诲嵆椤㈡瑩鏌ㄩ姘闂佹寧绻傛鎼佸几閻斿吋鐓曢悗锝庡亜婵秹鏌＄仦鍓ф创妤犵偛娲、妤佹媴閸欏浜炴繝? {exc}")
        return docs

    def _init_document_index(self, docs: Optional[List[Any]] = None) -> None:
        if docs is not None:
            self._document_index = list(docs)
        elif not self._document_index and self.vs is not None:
            self._document_index = self._extract_docs_from_loaded_store()

        if self.cfg["keyword_search_enabled"] and self._document_index:
            try:
                self._calculate_tf_idf()
            except Exception as exc:
                logger.warning(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾圭€瑰嫭鍣磋ぐ鎺戠倞鐟滃繘寮抽敃鍌涚厱妞ゎ厽鍨垫禍婵嬫煕濞嗗繒绠抽柍褜鍓濋～澶娒洪弽顬℃椽鏁冮崒銈嗘櫇婵炲濮撮鍡涙偂韫囨稒鐓曟い鎰剁悼缁犳牕顭跨捄鍝勵仾闁靛洤瀚伴、鏇㈡晲閸℃瑯妲归梻浣告惈閻鎹㈠┑鍡欐殾闁割偅娲栧敮闂侀潧锛忛崒婊勶紡闂?TF-IDF 濠电姷鏁告慨鐑藉极閸涘﹥鍙忛柣鎴ｆ閺嬩線鏌涘☉姗堟敾闁告瑥绻愰湁闁稿繐鍚嬬紞鎴︽煕閵娿儱鈧湱鎹㈠☉姗嗗晠妞ゆ棁宕甸惄搴ｇ磼閻愵剙鍔ゆ繛纭风節瀵濡堕崥銈呮贡閳ь剨缍嗛崑鍛存偟閺冨牊鈷戦柟鑲╁仜婵＄晫绱掔拠鑼ⅵ鐎殿喛顕ч埥澶愬閻樻鍟嬫繝寰锋澘鈧劙宕戦幘缁樺€垫慨妯煎帶婢у鈧? {exc}")

    def _calculate_tf_idf(self) -> None:
        if not self._document_index:
            return
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [doc.page_content for doc in self._document_index if getattr(doc, "page_content", None)]
        if not texts:
            return
        self._tfidf_vectorizer = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b")
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)

    def _get_persist_dirs(self) -> tuple[str, str]:
        abs_dir = os.path.abspath(self.cfg["persist_dir"])
        cwd = os.path.abspath(os.getcwd())
        if abs_dir.startswith(cwd):
            rel_dir = os.path.relpath(abs_dir, cwd)
        else:
            rel_dir = abs_dir
        return abs_dir, rel_dir

    def _save_vector_store(self) -> None:
        if self.vs is None:
            return
        abs_dir, rel_dir = self._get_persist_dirs()
        os.makedirs(abs_dir, exist_ok=True)
        os.makedirs(rel_dir, exist_ok=True)
        self.vs.save_local(rel_dir)

    def load_or_build(self) -> bool:
        if not self._lock.acquire(blocking=False):
            logger.info("Waiting for vector store build in another thread")
            self._lock.acquire()

        try:
            self.reload_config()
            if self.imports is None:
                logger.error("langchain dependencies are unavailable; cannot initialize vector store")
                return False

            if self.vs is not None:
                return True

            self._ensure_embeddings()
            if self.embeddings is None:
                logger.error("Embedding model is unavailable; cannot load or build vector store")
                return False

            persist_dir, persist_dir_rel = self._get_persist_dirs()
            signatures = self._collect_document_signatures()
            build_meta = self._load_build_metadata() or {}
            should_rebuild = False

            if not os.path.isdir(persist_dir):
                logger.info(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾剧懓顪冪€ｎ亝鎹ｉ柣顓炴閵嗘帒顫濋敐鍛婵°倗濮烽崑娑⑺囬悽绋挎瀬鐎广儱顦粈瀣亜閹板墎鎮奸柍褜鍓氭繛濠囧箖濡ゅ懐宓侀柛顭戝枛婵骸顪冮妶蹇撶槣闁搞劏娉涢悾鐤亹閹烘垹锛滈梺缁樺姇椤曨參宕㈤柆宥嗗仭婵犲﹤鍟撮崣鍕煏閸℃鏆ｇ€规洏鍔庨埀顒佺⊕鑿ら柟鐤嚙椤啴濡堕崱娆忊拡闂佺顑囬崑銈夊箖閿熺姵鍋勯柛蹇氬亹閸樼敻鏌ｆ惔锝嗘毄妞ゎ厼鐗撻崺鈧い鎺嶈兌婢у灚顨ラ悙宸剶闁轰礁鍟撮崺鈧い鎺戝€搁ˉ姘舵煕瑜庨〃鍡涙偂閻斿吋鐓涢柛灞炬皑娴犮垽鏌熼钘夌伌闁哄矉缍侀獮妯尖偓闈涙啞閸ｄ即姊洪崫鍕伇闁哥姵鎹囬獮澶愬箻椤旇偐顦板銈嗗姂閸╁嫬顭囬幘缁樷拻濞达綀娅ｇ敮娑㈡煙缁嬫鐓肩€规洘绻傝灃闁告劏鏅╁鐔兼⒑閸︻厼鍔嬫い銊ユ噽缁鈽夊▎宥勭盎闂佸湱鍋撻〃鍛偓姘煎櫍瀵娊顢橀姀鈾€鎷绘繛杈剧悼閸庛倝宕甸埀顒勬倵濞堝灝鏋涢柣鏍с偢閻涱噣宕橀鑲╋紲闂佺粯鍔樼亸娆撴偩濞差亝鍋℃繝濠傚閻撱儵鏌ｉ敐澶嬫暠缂佽桨绮欏畷銊︾節閸屾瑧鐩庨梻鍌欒兌缁垶鎳濇ィ鍐炬晪妞ゆ挶鍨诲畵渚€鏌曡箛瀣伄缁惧彞绮欓弻娑氫沪閹规劕顥濋梺閫炲苯澧伴柛蹇旓耿楠炲啴鎮欓悜妯绘珖闂佺鏈彜闁稿鎹囧畷姗€顢欓悡搴ｇ崺婵犳鍠楃缓鍧楀磹閺囩伝娲敇閵忕姾鎽曢梺绯曞墲椤ㄥ繘宕ョ€ｎ喗鐓曟繛鎴濆船閻忕姴鈹戦鍡涙濞ｅ洤锕幃娆擃敂閸曘劌浜鹃柡宥庡亝閺嗘粌鈹戦悩鍙夊闁搞倕瀚伴弻娑㈩敃閿濆棛顦ラ梺缁樻尨閸嬫捇姊绘担鍛婃儓婵炲眰鍨藉畷鐟懊洪鍛簵闂侀潧顧€闂勫嫰宕ｈ箛鏃€鍙忔俊銈傚亾婵☆偅顨婇悰顔嘉熼懖鈺冿紲闂佹娊鏁崑鎾绘煕鐎ｎ偅宕屾慨濠勭帛缁楃喖鍩€椤掑嫬鐒垫い鎺戝€告禒婊堟煠濞茶鐏￠柡鍛閳ь剚绋掕彠濞存粍绮撻弻鏇＄疀婵犲倸鈷夊銈冨劚閻楀繘鍩€椤掑喚娼愭繛鍙夌墵婵″爼骞栨担鍝ョ暫闂侀潧绻堥崐鏇烆啅濠靛棌鏀介柣妯诲絻椤忊晝鈧偣鍊栧钘夘潖婵犳艾纾兼繛鍡樺姉閵堟澘顪冮妶鍡樿偁闁告侗鍠栨惔濠囨⒑閸︻厼鍔嬫い銊ユ噽婢规洘绻濆顓犲幗闂佸搫鍟崐鍝ユ暜閸洘鐓熼柕鍫濇搐娴滈箖鏌曢崶褍顏鐐叉喘婵℃悂鏁冮埀顒€鈻嶅鑸碘拺闁告繂瀚烽崕鎰叏婵犲倹璐￠柟骞垮灩閳藉顫滈崱妤佹珚婵＄偑鍊栧Λ浣哥暦閻㈢绀勭憸鐗堝笚閳锋垹绱撴担鑲℃垹浜告导瀛樼厽闁绘梹绻傞幊蹇涳綖閺囥垺鐓欓柣鎴灻▓楣冩煛閸愶絽浜鹃梺鐟板槻閹虫劙骞冩导鏉戠労闁告劗鍋撶紞灞剧節閻㈤潧啸妞わ綀妫勫嵄闁告稒娼欑壕濠氭煙閻戞﹩娈旈柣顓燁殜閺屾盯骞囬悽闈涘弗闂佹悶鍊栧ú鐔煎蓟閺囩喓绠鹃柣鎰靛墯閻濇洟鏌ｆ惔锝嗘毄濠殿喗鎸抽垾锕傚锤濡も偓缁€鍌氼熆鐠轰警鍎愰柣婵囨礋濮婃椽鎮滈埡鍌涚彟闂佹悶鍔忓▔娑㈩敋閿濆鏁冮柨鏇楀亾缂佺姵鐩弻鈩冨緞婵犲嫪铏庨梺鍏兼緲閻忔繈鍩為幋锔藉€烽柛娆忣樈濡偛鈹戦悙鑼憼闁煎綊绠栭垾锕傚炊椤掍胶鍙嗛柣搴岛閺呮繄绮诲鑸碘拺缂備焦锚婵箑霉濠婂嫮鐭掗柛鈹惧亾濡炪倖甯婇懗鑸垫櫠閻㈢鍋撶憴鍕缂佽鐗嗛锝夊箻椤旂⒈娼婇梺缁樕戦悥鐘诲川椤栨粣绱查梺璇插嚱缂嶅棝宕戦幒妤€纾块柕澶嗘櫆閻撴洟鏌曢崼婵囶棡缂佲偓鐎ｎ喗鐓涚€光偓閳ь剟宕伴弽褜娼栭柤濮愬€愰崑鎾绘濞戞﹩妫屽┑鈥虫▕閸犳氨妲愰幘瀛樺闁圭粯甯婃竟鏇㈡⒑鐠囨彃鍤辩紓宥呮瀹曟垶鎷呴崷顓ф锤濠电姴锕ら悧濠囨偂濞嗘挻鐓欏ù锝呭暞閻濐亝绻涢崨顓熷殗闁哄苯绉烽¨渚€鏌涢幘鏉戝摵闁靛棗鍟村畷鍗炍旀繝鍌涱唫? {persist_dir}")
                should_rebuild = True
            elif not build_meta:
                logger.info("Build metadata not found; rebuilding vector store")
                should_rebuild = True
            elif self._documents_changed(build_meta.get("documents", {}), signatures):
                logger.info("Knowledge base documents changed; rebuilding vector store")
                should_rebuild = True

            if not should_rebuild:
                try:
                    load_path = persist_dir_rel if os.path.isdir(persist_dir_rel) else persist_dir
                    self.vs = self.imports["FAISS"].load_local(
                        load_path,
                        self.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    self._init_document_index()
                    if (
                        build_meta.get("signature_scope") != "knowledge_sources_v2"
                        or build_meta.get("documents") != signatures
                    ):
                        self._save_build_metadata(signatures)
                    logger.info("Loaded existing vector store successfully")
                    return True
                except Exception as exc:
                    logger.warning(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾剧懓顪冪€ｎ亝鎹ｉ柣顓炴閵嗘帒顫濋敐鍛婵°倗濮烽崑娑⑺囬悽绋垮瀭濡わ絽鍟粻娑樏归敐鍛础缂佸鑳剁槐鎾诲磼濮橆兘鍋撻悜鑺ュ€块柨鏇氱劍閹冲矂姊绘担鍛婂暈闁荤喆鍎抽幑銏狀潨閳ь剙顕ｆ繝姘亜缁炬媽椴搁弲顒勬⒑閹稿海绠撴繛璇х到鏁堟俊銈呮噺閸嬧剝绻濇繝鍌涘櫣妞わ絽銈搁幃浠嬵敍濞戞ɑ璇為梺璇″枟閻燂妇鎹㈠┑瀣妞ゆ巻鍋撻柡瀣灥閳规垿鎮╃拠褍浼愰梺缁橆殔濡繈骞冮崸妤婃晪闁逞屽墴瀵鈽夊Ο閿嬵潔濠碘槅鍨伴崥瀣箟婵傚憡鈷戠紓浣股戠亸鐗堢箾閸欏鐒介柟骞垮灩閳藉濮€閻樿尪鈧灝鈹戞幊閸婃捇鎳楅崼鏇樷偓鍛村级鎼存挻鏂€闂佹枼鏅涢崯顐ゅ緤缂佹绡€闁逞屽墯濞煎繘鍩￠崒銈嗙稐闂備礁婀遍崕銈夈€冮崱娑樼；闁规壆澧楅悡鏇㈡煙闁箑澧紒鐙欏洦鍋ｉ柛銉㈡櫅娴犺鲸鎱ㄦ繝鍐┿仢鐎规洦鍋婂畷鐔碱敆娴ｇ懓顏扮紓鍌氬€风粈渚€顢栭崱娑樺瀭闁秆勵殕閺呮煡鏌ｉ幇顓犮偞闁衡偓娴犲鍊甸柨婵嗘噽娴犳盯鏌￠崪鍐М婵﹨娅ｇ槐鎺懳熼崫鍕垫綋闂備礁鎲￠崺鍐磻閹惧墎纾藉ù锝勭矙閸濊櫣绱掔拠鑼妞ゎ偄绻掔槐鎺懳熼懖鈺傚殞闂備線鈧偛鑻晶顔姐亜椤撶偞鍠橀柟绛圭節婵″爼宕ㄩ婊呮憣闂傚倷鑳堕～瀣礋閸偆鏆ョ紓浣哄亾閸庡啿顭囬敓鐘茶摕闁哄洢鍨归柋鍥ㄧ節閸偄濮堢粭鎴犵磽閸屾瑨鍏屽┑顕€娼ч悾婵嬪箹娴ｆ瓕鎽曢梺璺ㄥ枔婵绮堥崘顔界厪濠电倯鍐ㄦ灓闁哄棭鍋嗙槐鎾诲磼濞嗘帒鍘℃繝娈垮枟閹告娊鍨鹃敃鍌涘€婚柦妯侯槼閹芥洟姊洪幐搴ｇ畵妞わ富鍨崇划璇测槈濞嗗秳绨婚梺瑙勫礃濞夋稒绂掕闇夋繝濞惧亾闁告鍟块～蹇撁洪鍕炊闂佸憡娲栭悘姘跺煝閹剧粯鈷戦柛娑橈功閹冲啴鏌涢悢鍛婄稇妞ゆ洩绲剧换婵嗩潩椤撶喐鐝冲┑鐘灱濞夋盯顢栭崨顔绢洸鐟滅増甯楅埛鎴︽煕濞戞﹫宸ュ┑顔肩墢閹叉悂寮堕崹顔芥闂佹寧绻勯崑銈呯暦閸洦鏁嗗ù锝呭级鐎氳偐绱撻崒娆戭槮妞ゆ垵妫涢埀顒傜懗閸パ嗗煘婵犮垼娉涙径鍥磻閹捐埖鍠嗛柛鏇ㄥ墰椤︽澘鈹戦悙鏉垮皟闁告劑鍔庨妶顐︽⒒閸屾瑧顦︾紓宥咃躬瀹曟垶绻濋崶褏顦у┑顔姐仜閸嬫挾鈧鍠撻崝宥囩矉閹烘柡鍋撻敐搴′簽闁告ê宕—鍐Χ閸℃衼缂備焦褰冨锟犲箚鐏炵晫鏆嗛柛鏇ㄥ墰閸橀亶姊虹憴鍕姢鐎规洦鍓氶幈銊╁磼濞戞瑥寮挎繝鐢靛Т閹冲繘顢旈悩缁樼厱闁哄秲鍔庢晶鐢碘偓娈垮枟閹告娊骞冨▎寰濆湱鈧綆鍋勯悵鍓佺磽閸屾艾鈧悂宕愭搴ｇ焼濞撴埃鍋撴い銏＄墵瀹曞崬鈽夊Ο铏光偓顓㈡⒑濮瑰洤鐏い顓炵墦閹€斥攽閸艾浜炬鐐茬仢閸旀碍銇勯敂璇茬仸闁挎繄鍋ゅ畷銊р偓娑欘焽閸橆亪姊洪崜鎻掍簼缂佽鍟磋棢闁绘绮悡鏇熶繆椤栨粎甯涢柛濠冨姈閹便劍绻濋崨顕呬哗缂備緡鍠楅悷鈺呯嵁閹邦厽鍎熼柕蹇娾偓鎰佷紗缂傚倸鍊搁崐椋庢閿熺姴绐楅柡宥庡幗閸庢鏌涘畝鈧崑娑滅箽闂備胶顭堥張顒傜矙閹剧粯鈷旈柛鏇ㄥ灡閻撴洘绻涢幋鐑囧叕闁告梻鍠栭弻娑樜熼崷顓犵厯闂佸搫鐬奸崰鏍х暦濡ゅ懎绀傞柣鎾抽婢规帡姊绘担鍛婃喐濠殿喚鏁婚幃褔鎮╃拠鑼暫濠殿喗銇涢崑鎾绘煙椤旇娅婇柟顔界懇楠炲骞掗弮鍌涙櫒? {exc}")

            docs = self._collect_documents()
            if not docs:
                logger.warning("No eligible documents found for vector store build")
                return False

            logger.info(f"Collected {len(docs)} source documents; start splitting")
            split_docs = self._split_documents(docs)
            if not split_docs:
                logger.warning("No document chunks were produced after splitting")
                return False

            logger.info(f"Document splitting finished; got {len(split_docs)} chunks and start building vector store")
            self.vs = self.imports["FAISS"].from_documents(split_docs, self.embeddings)
            self._save_vector_store()
            self._init_document_index(split_docs)
            self._save_build_metadata(signatures)
            logger.info("Vector store build completed")
            return True
        except Exception as exc:
            logger.error(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾剧懓顪冪€ｎ亝鎹ｉ柣顓炴閵嗘帒顫濋敐鍛婵°倗濮烽崑娑⑺囬悽绋垮瀭濡わ絽鍟粻娑樏归敐鍛础缂佸鑳剁槐鎾诲磼濮橆兘鍋撻悜鑺ュ€块柨鏇氱劍閹冲矂姊绘担鍛婂暈闁荤喆鍎抽幑銏狀潨閳ь剙顕ｆ繝姘亜缁炬媽椴搁弲顒勬⒑閹稿海绠撴繛璇х到鏁堟俊銈呮噺閸嬧剝绻濇繝鍌涘櫣妞わ絽銈搁幃浠嬵敍濞戞ɑ璇為梺璇″枟閻燂妇鎹㈠┑瀣闁挎棁娉曢惄搴ㄦ⒒娴ｇ儤鍤€闁圭⒈鍋勮灋婵°倓鑳堕々鏌ユ煕閹炬鎳愰敍婵囩箾鏉堝墽鍒伴柟纰卞亰椤㈡捇骞橀崜浣猴紳閻庡箍鍎辩€氼喚绮ｉ弮鍌楀亾濞堝灝鏋涙い顓犲厴瀹曡銈ｉ崘銊х杸婵炶揪绲介幉锟犲磹椤栫偞鈷掑ù锝呮啞閸熺偤鏌涢弮鈧崹鍨暦濠靛棛鏆嗛柛鏇樺妷閸嬫捇寮介鐔蜂壕闁挎繂楠搁弸鐔兼煃闁垮鐏存慨濠冩そ椤㈡洟濡堕崨顒傛崟闂備礁鍚嬪鍧楀垂闁秴鐤鹃柤鍝ユ暩缁♀偓濠殿喗锕╅崢楣冨储闁秵鈷戦梻鍫熶緱閻掗箖鏌涙繝鍐╃闁伙絿鏁诲畷鍗炍熼梹鎰泿闂備線娼х欢鐢稿Χ閸滀焦顥ら梻鍌欑閻ゅ洭顢氳閺屽﹪鏁愭径灞界ウ濠殿喗銇涢崑鎾垛偓瑙勬礃椤ㄥ﹤鐣烽妸鈺婃晬婵犲﹤鍟犻崑鎾绘倷閻戞ǚ鎷婚梺绋挎湰閻熝囁囬敃鍌涚厵婵炴潙顑傞崑鎾舵惥娴ｈ銇濋柡浣稿暣瀹曟帒顫濇鏍ㄐゅ┑鐘垫暩閸嬫稑螞濞嗘挸绠伴柟闂寸劍閸婂潡鏌涢…鎴濅簴濞存粍绮撻弻鐔煎传閸曨剦妫炴繛瀛樼矋閸庢娊鈥旈崘顔嘉ч幖绮光偓宕囶啇闂備焦鎮堕崹娲偂閿熺姵鏅查柣鎰閻も偓闂佺懓褰為悞锕€顪冩禒瀣ㄢ偓渚€寮崼婵堫槹濡炪倖鎸嗛崘鈺傛緫闂傚倸鍊搁崐椋庣矆娴ｉ潻鑰块梺顒€绉寸壕鍧楁煏閸繃濯兼繛鍏肩墪閳规垿鎮╅崘鎻掓瘓闂佽桨绀侀敃顏堝蓟濞戙垹唯闁靛繆鍓濋悵鏍磼閻愵剙绀冩俊顐㈠濠€渚€姊洪幐搴ｇ畵妞わ附婢橀埥澶庮槻闁宠鍨块、娆撴偂鎼存ê浜鹃柟闂寸劍缁犳帗绻濋悽闈浶㈤柨鏇樺€曡灋婵炲棙鍨堕崗婊堟煟閹伴潧鍘靛ù婊勭矒閺岋繝宕堕張鐢垫晼缂備礁顦介崰妤呭Φ閸曨垰顫呴柨娑樺閸ｄ即姊虹拠鈥虫灍闁荤啿鏅犻妴浣肝旈崨顓犲姦濡炪倖甯掗崐鎼佀夊顓滀簻闁哄稁鍋勬禒锕傛煃闁垮鐏撮柡灞剧☉閳藉顫滈崼婵嗩潬缂傚倷鑳舵慨鐢稿箲閸ヮ剙绠栨俊銈傚亾妞ゎ偅绻堥幃鈩冩償閵忣澀閭梻鍌欐缁鳖喚寰婇崜褎宕查柛宀€鍋涢悡鈥愁熆閼搁潧濮囨い顐㈡嚇閺屽秹鏌ㄩ姘闂備胶鎳撻崲鏌ュ箹椤愶箑鐓橀柟杈惧瘜閺佸﹪鏌ｉ幇闈涘婵炲牆鑻埞鎴︽倷閼碱剚鍕鹃梺绋匡攻缁诲牓鐛崘顭戞建闁逞屽墴閵嗕礁顫濈捄铏瑰姦濡炪倖甯掔€氼剟鎷戦悢鍝ョ闁瑰瓨鐟ラ悘顏堟煕鐎ｎ偄濮嶉柡灞剧☉閳诲氦绠涢敐鍠把囨倵鐟欏嫭绀€闁哥喎娼￠崺鐐哄箣閿曗偓閻忔娊鏌涢幇顖氱毢濡ょ姴娲娲川婵犲啠鎷归梺鑽ゅ暱閺呯娀骞冩导鎼晩缁炬媽椴稿娲⒑闁偛鑻晶瀛橆殽閻愯宸ユい顓滃姂瀹曠厧鈹戦崱娆屽亾椤撶儐娓婚柕鍫濇婵呯磽瀹ュ懏顥㈢€? {exc}", exc_info=True)
            return False
        finally:
            self._lock.release()

    def ingest_texts(
        self,
        texts: List[str],
        subject: str = "general",
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        if self.imports is None or not texts:
            return 0

        self.reload_config()
        self._ensure_embeddings()
        if self.embeddings is None:
            logger.error("Embedding model is unavailable; cannot ingest texts into vector store")
            return 0

        Document = self.imports["Document"]
        raw_docs = []
        for index, text in enumerate(texts):
            meta = metadata[index] if metadata and index < len(metadata) else {}
            cleaned = self._clean_text(str(text or ""))
            if not cleaned or self._is_low_quality_text(cleaned):
                continue
            raw_docs.append(
                Document(
                    page_content=cleaned,
                    metadata={
                        "source": meta.get("source", "manual"),
                        "subject": meta.get("subject", subject),
                        **meta,
                    },
                )
            )

        split_docs = self._split_documents(raw_docs)
        if not split_docs:
            return 0

        try:
            if self.vs is None:
                if not self.load_or_build():
                    self.vs = self.imports["FAISS"].from_documents(split_docs, self.embeddings)
                    self._document_index = []
            else:
                self.vs.add_documents(split_docs)

            if self.vs is not None and not getattr(self.vs, "index_to_docstore_id", None):
                self.vs = self.imports["FAISS"].from_documents(split_docs, self.embeddings)

            self._document_index.extend(split_docs)
            self._init_document_index(self._document_index)
            self._save_vector_store()
            return len(split_docs)
        except Exception as exc:
            logger.error(f"闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾剧懓顪冪€ｎ亝鎹ｉ柣顓炴閵嗘帒顫濋敐鍛婵°倗濮烽崑娑⑺囬悽绋挎瀬闁瑰墽绮崑鎰版煠绾板崬澧绘俊鑼厴濮婄粯鎷呴崨濠冨創缂備礁顑勭欢姘暦閵忥紕顩烽悗锝庝簽閿涙盯姊虹捄銊ユ灁濠殿喚鏁婚幃陇绠涢幙鍐數闁荤姾娅ｉ崕銈壦夐悙瀵哥闁告侗鍘介崵鍥煛瀹€瀣М妞ゃ垺锕㈤幃銈嗘媴閸︻厽婢掗梻鍌欑閹碱偊寮甸鍕剮妞ゆ牜鍋涢弰銉╂煕閺囥劌鐏犵紒鈧崘鈹夸簻闊洤娴烽崝宥嗐亜椤愩垻效婵﹥妞介幊锟犲Χ閸涱喚鈧箖姊洪懡銈呮灆濞存粠鍓涢崚鎺撶節濮橆剛顔掗柣鐘叉穿鐏忣亪骞楅弴銏♀拺缂備焦蓱閻撱儵鏌熺拠褏纾跨紒杈╁仱瀹曞爼顢楁担鍝勫箞闂備礁婀遍崑鎾汇€冮崨鏉戠疇闊洦鎸撮弸鏃堟煕濞戞﹫鍔熺紒鈾€鍋撻梻浣圭湽閸ㄨ棄顭囪缁傛帒顭ㄩ崼鐔哄幍闂佸憡鍔樼亸娆撴倿閸濄儮鍋撶憴鍕闁告梹娲熼崺鐐哄箣閻橆偄浜鹃柨婵嗛婢ь喚绱掓笟濠勭暤婵﹥妞藉畷顐﹀礋椤撴稒鐎遍梻浣侯焾椤戝倿宕戦幘缁樷拺缂佸顑欓崕鎰版煙閸涘﹥鍊愰柛鈹垮劜瀵板嫭绻涢悙顒傗偓娲⒑閹勭闁稿瀚伴幃姗€鍩℃导杈╂嚀椤劑宕熼鈧埀顒佸姈閵囧嫰濮€閳藉懓鈧灝鈹戦鐟颁壕闂備焦瀵х粙鎴犫偓姘煎墰閺侇喗銈ｉ崘鈹炬嫽婵炶揪缍€椤宕戦悩缁樼厱閹兼惌鍠栭悘锔锯偓瑙勬礃缁诲倿鎮惧┑瀣妞ゆ巻鍋撻柍褜鍓欓悥濂稿蓟瀹ュ浼犻柛鏇ㄥ墮濞呫倝姊? {exc}", exc_info=True)
            return 0

    def keyword_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.cfg["keyword_search_enabled"] or not self._document_index or not query:
            return []

        results: List[Dict[str, Any]] = []
        if self._tfidf_vectorizer is not None and self._tfidf_matrix is not None:
            try:
                from sklearn.metrics.pairwise import cosine_similarity

                query_vec = self._tfidf_vectorizer.transform([query])
                similarities = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
                indices = [i for i, score in enumerate(similarities) if score > 0]
                indices = sorted(indices, key=lambda i: similarities[i], reverse=True)[:top_k]
                for index in indices:
                    results.append(
                        {"doc": self._document_index[index], "score": float(similarities[index])}
                    )
                return results
            except Exception as exc:
                logger.warning(f"TF-IDF ????????????????: {exc}")

        tokens = [t for t in re.findall(r"[濠电姷鏁告慨鐑藉极閸涘﹥鍙忛柣鎴ｆ閺嬩線鏌熼梻瀵割槮缁炬儳顭烽弻锝夊箛椤掍焦鍎撻梺鎼炲妼閸婂潡寮诲☉銏╂晝闁挎繂妫涢ˇ銉х磽娴ｅ搫小闁告濞婂濠氭偄閾忓湱锛滈梺闈涚箳婵敻鎮橀崼銏㈢＜?濠电姷鏁告慨鐑藉极閸涘﹥鍙忛柣銏犲閺佸﹪鏌″搴″箹缂佹劖顨嗘穱濠囧Χ閸涱厽娈查梺鍝勬缁捇寮诲澶婁紶闁告洦鍓欏▍锝夋⒑缁嬭儻顫﹂柛鏃€鍨垮璇测槈閵忕姷顔掑┑掳鍊愰崑鎾绘煙閾忣偄濮嶉柟?|[A-Za-z0-9_]+", query.lower()) if len(t) >= 2]
        if not tokens:
            return []

        scored = []
        for doc in self._document_index:
            content = getattr(doc, "page_content", "").lower()
            score = sum(content.count(token) for token in tokens)
            if score > 0:
                scored.append((doc, float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [{"doc": doc, "score": score} for doc, score in scored[:top_k]]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 4,
        subject: Optional[str] = None,
        k: Optional[int] = None,
    ) -> List[Any]:
        top_k = k or top_k
        if self.vs is None:
            return [item["doc"] for item in self.keyword_search(query, top_k)]

        candidate_multiplier = max(2, int(self.cfg.get("hybrid_candidate_multiplier", 4)))
        candidate_k = max(top_k * candidate_multiplier, top_k)
        vector_results = self.vs.similarity_search_with_score(query, k=candidate_k)
        keyword_results = self.keyword_search(query, top_k=candidate_k)

        vector_weight = float(self.cfg.get("vector_weight", 0.7))
        keyword_weight = float(self.cfg.get("keyword_weight", 0.3))
        subject_match_boost = float(self.cfg.get("subject_match_boost", 0.15))
        subject_mismatch_penalty = float(self.cfg.get("subject_mismatch_penalty", 0.18))
        query_match_boost = float(self.cfg.get("query_match_boost", 0.25))
        primary_doc_boost = float(self.cfg.get("primary_doc_boost", 0.05))
        supplementary_doc_penalty = float(self.cfg.get("supplementary_doc_penalty", 0.03))
        query_terms = [term for term in re.findall(r"[濠电姷鏁告慨鐑藉极閸涘﹥鍙忛柣鎴ｆ閺嬩線鏌熼梻瀵割槮缁炬儳顭烽弻锝夊箛椤掍焦鍎撻梺鎼炲妼閸婂潡寮诲☉銏╂晝闁挎繂妫涢ˇ銉х磽娴ｅ搫小闁告濞婂濠氭偄閾忓湱锛滈梺闈涚箳婵敻鎮橀崼銏㈢＜?濠电姷鏁告慨鐑藉极閸涘﹥鍙忛柣銏犲閺佸﹪鏌″搴″箹缂佹劖顨嗘穱濠囧Χ閸涱厽娈查梺鍝勬缁捇寮诲澶婁紶闁告洦鍓欏▍锝夋⒑缁嬭儻顫﹂柛鏃€鍨垮璇测槈閵忕姷顔掑┑掳鍊愰崑鎾绘煙閾忣偄濮嶉柟?|[A-Za-z0-9_]+", query.lower()) if len(term) >= 2]

        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Any] = {}
        constant = 60

        for rank, (doc, _) in enumerate(vector_results, start=1):
            doc_id = getattr(doc, "page_content", "") or str(rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + vector_weight * (1.0 / (rank + constant))
            doc_map[doc_id] = doc

        for rank, item in enumerate(keyword_results, start=1):
            doc = item["doc"]
            doc_id = getattr(doc, "page_content", "") or str(rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + keyword_weight * (1.0 / (rank + constant))
            doc_map[doc_id] = doc

        for doc_id, doc in list(doc_map.items()):
            metadata = getattr(doc, "metadata", {}) or {}
            content = getattr(doc, "page_content", "").lower()
            if subject and subject != "??" and subject != "通用":
                doc_subject = self._doc_subject_for_ranking(doc)
                if doc_subject == subject:
                    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + subject_match_boost
                elif doc_subject and doc_subject != "通用":
                    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) - subject_mismatch_penalty
            if query_terms:
                overlap = sum(1 for term in query_terms if term in content)
                if overlap:
                    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + min(
                        query_match_boost, overlap * 0.03
                    )

            doc_role = self._infer_doc_role(str(metadata.get("file_name") or ""))
            if doc_role == "primary":
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + primary_doc_boost
            elif doc_role == "supplementary":
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) - supplementary_doc_penalty
            elif doc_role == "helper":
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) - (supplementary_doc_penalty * 2)

            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + self._score_doc_source_bias(
                doc, query_terms
            )

        ranked_ids = sorted(rrf_scores, key=lambda item: rrf_scores[item], reverse=True)
        ranked_docs = [doc_map[item] for item in ranked_ids]
        docs = self._select_diverse_documents(ranked_docs, top_k)
        return self.rerank(query, docs)

    def rerank(self, query: str, documents: List[Any]) -> List[Any]:
        if not documents or not self.cfg.get("enable_rerank", False):
            return documents

        try:
            from sentence_transformers import CrossEncoder

            model_name = config.get("rag.rerank_model", "BAAI/bge-reranker-base")
            if not hasattr(self, "_reranker"):
                logger.info(f"??????: {model_name}")
                self._reranker = CrossEncoder(model_name)

            pairs = [[query, doc.page_content] for doc in documents]
            scores = self._reranker.predict(pairs)
            ranked = sorted(zip(documents, scores), key=lambda item: item[1], reverse=True)
            return [doc for doc, _ in ranked]
        except Exception as exc:
            logger.warning(f"????????????: {exc}")
            return documents

    def rebuild(self) -> bool:
        persist_dir = os.path.abspath(self.cfg["persist_dir"])
        if os.path.isdir(persist_dir):
            import shutil

            shutil.rmtree(persist_dir)
        self.vs = None
        self._document_index = []
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        return self.load_or_build()

    def get_document_stats(self) -> Dict[str, Any]:
        meta = self._load_build_metadata() or {}
        docs_meta = self._normalize_signature_snapshot(
            meta.get("documents", {}) if isinstance(meta.get("documents"), dict) else {}
        )
        total_documents = len(docs_meta)

        total_chunks = 0
        try:
            if self.vs is not None and hasattr(self.vs, "index_to_docstore_id"):
                total_chunks = len(getattr(self.vs, "index_to_docstore_id"))
        except Exception:
            total_chunks = 0

        if not total_chunks:
            total_chunks = len(self._document_index)

        subjects = set()
        for doc in self._document_index:
            metadata = getattr(doc, "metadata", {}) or {}
            subject = metadata.get("subject")
            if subject:
                subjects.add(str(subject))

        return {
            "built_at": meta.get("built_at"),
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "subject_count": len(subjects),
        }


