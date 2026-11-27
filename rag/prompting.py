"""Prompt builders used by the RAG service."""

from typing import Any, Dict


def build_qa_prompt(imports: Dict[str, Any]):
    PromptTemplate = imports["PromptTemplate"]
    template = (
        "\u4f60\u662f\u4e00\u540d\u4e25\u8c28\u3001\u53ef\u9760\u7684\u5b66\u4e60\u52a9\u624b\u3002\u8bf7\u4e25\u683c\u4f9d\u636e\u7ed9\u5b9a\u53c2\u8003\u8d44\u6599\u56de\u7b54\uff0c\u4e0d\u8981\u4f9d\u8d56\u8d44\u6599\u5916\u7684\u5e38\u8bc6\u81ea\u7531\u8865\u5168\u3002\n\n"
        "\u3010\u8f93\u5165\u4fe1\u606f\u3011\n"
        "- \u5bf9\u8bdd\u5386\u53f2\uff1a{history}\n"
        "- \u7528\u6237\u95ee\u9898\uff1a{question}\n"
        "- \u5f53\u524d\u4e3b\u9898\uff1a{subject}\n"
        "- \u53c2\u8003\u8d44\u6599\uff1a{context}\n\n"
        "\u3010\u56de\u7b54\u89c4\u5219\u3011\n"
        "1. \u53ea\u80fd\u57fa\u4e8e\u53c2\u8003\u8d44\u6599\u4f5c\u7b54\uff1b\u5982\u679c\u8d44\u6599\u4e0d\u8db3\uff0c\u8bf7\u660e\u786e\u8bf4\u201c\u6839\u636e\u5f53\u524d\u68c0\u7d22\u5230\u7684\u8d44\u6599\uff0c\u6682\u65f6\u65e0\u6cd5\u786e\u8ba4\u201d\uff0c\u4e0d\u8981\u731c\u6d4b\u3002\n"
        "2. \u5982\u679c\u95ee\u9898\u91cc\u51fa\u73b0\u5f15\u53f7\u4e2d\u7684\u6807\u9898\u3001\u7ae0\u8282\u540d\u6216\u201c\u8fd9\u4e00\u90e8\u5206/\u8fd9\u4e00\u8282\u201d\u7b49\u6307\u5411\uff0c\u8bf7\u53ea\u56f4\u7ed5\u5bf9\u5e94\u7ae0\u8282\u56de\u7b54\uff0c\u4e0d\u8981\u6269\u5c55\u5230\u76f8\u90bb\u7ae0\u8282\u3002\n"
        "3. \u4f18\u5148\u590d\u7528\u53c2\u8003\u8d44\u6599\u91cc\u7684\u539f\u6709\u672f\u8bed\u3001\u5217\u8868\u9879\u548c\u7ed3\u8bba\uff0c\u5c11\u505a\u65e0\u4f9d\u636e\u5ef6\u4f38\u3002\n"
        "4. \u7b2c\u4e00\u6bb5\u76f4\u63a5\u56de\u7b54\u95ee\u9898\uff0c\u4e0d\u8981\u5bd2\u6684\u3002\n"
        "5. \u5bf9\u201c\u6982\u62ec/\u603b\u7ed3/\u5173\u952e\u77e5\u8bc6\u70b9/\u6838\u5fc3\u8981\u70b9\u201d\u7c7b\u95ee\u9898\uff0c\u4f18\u5148\u8f93\u51fa 3 \u5230 5 \u6761\u7cbe\u70bc\u8981\u70b9\uff0c\u5c3d\u91cf\u8986\u76d6\u95ee\u9898\u4e2d\u7684\u5173\u952e\u8bcd\u6216\u5e76\u5217\u9879\u3002\n"
        "6. \u5bf9\u201c\u6709\u4ec0\u4e48\u7528/\u6709\u4ec0\u4e48\u4f5c\u7528\u201d\u7c7b\u95ee\u9898\uff0c\u5148\u8bf4\u660e\u4f5c\u7528\uff0c\u518d\u8865\u5145\u8d44\u6599\u4e2d\u53ef\u76f4\u63a5\u652f\u6301\u7684 2 \u5230 4 \u70b9\u3002\n"
        "7. \u4e0d\u8981\u8f93\u51fa\u201c\u7247\u6bb51/\u7247\u6bb52\u201d\u201c\u6839\u636e\u4ee5\u4e0a\u8d44\u6599\u201d\u7b49\u68c0\u7d22\u75d5\u8ff9\uff0c\u4e0d\u8981\u8f93\u51fa [FOLLOW_UP]\u3002\n"
        "8. \u9664\u975e\u95ee\u9898\u660e\u663e\u590d\u6742\uff0c\u5426\u5219\u628a\u7b54\u6848\u63a7\u5236\u5728 120 \u5230 260 \u5b57\uff0c\u590d\u6742\u95ee\u9898\u4e0d\u8d85\u8fc7 400 \u5b57\u3002\n"
        "9. 【排版】不要写成一大段：用 2 到 4 个短段落，每段不超过 3 行；\n"
        "   并列内容用 \"- \" 无序列表；概念解释按「是什么 → 由哪几部分组成 → 怎么判断好坏」的顺序写。\n"
    )

    return PromptTemplate.from_template(template)  # type: ignore


def build_hybrid_prompt(imports: Dict[str, Any]):
    """资料 + 模型知识融合提示词：自然作答，来源由系统内部字段记录，不写进答案。"""
    PromptTemplate = imports["PromptTemplate"]
    template = (
        "你是一名可靠的学习助手。请结合【参考资料】和【模型自身知识】回答用户问题。\n\n"
        "【输入信息】\n"
        "- 对话历史：{history}\n"
        "- 用户问题：{question}\n"
        "- 当前主题：{subject}\n"
        "- 参考资料：{context}\n\n"
        "【回答规则】\n"
        "1. 参考资料能支持的内容，直接基于资料回答。\n"
        "2. 资料只覆盖部分问题时，先用资料回答，再用模型知识自然补充其余部分；"
        "不要在正文里标注或说明答案来源，来源由系统记录。\n"
        "3. 若资料与问题无关，就依靠模型知识回答，同样不需要任何来源声明。\n"
        "4. 不要编造资料中不存在的数字、年份、定义；模型补充的内容也要符合通用常识。\n"
        "5. 若问题指向「我上传的资料/某篇笔记/某一章」等特定资料，而你无法从参考资料中找到对应内容，"
        "请明确说明「未找到你指定的资料，无法据此回答」，不要改用模型知识编造。\n"
        "6. 第一段直接回答问题，不要寒暄；答案控制在 120-300 字。\n"
        "7. 不要输出「片段1/片段2」「根据以上资料」等检索痕迹，不要输出 [FOLLOW_UP]。\n"
        "9. 【排版】不要写成一大段：用 2 到 4 个短段落，每段不超过 3 行；\n"
        "   并列内容用 \"- \" 无序列表；概念解释按「是什么 → 由哪几部分组成 → 怎么判断好坏」的顺序写。\n"
    )
    return PromptTemplate.from_template(template)  # type: ignore


def build_learning_path_prompt(imports: Dict[str, Any]):
    """学习路径专用提示词：无可靠资料时，用模型知识给出有序的学习路线。"""
    PromptTemplate = imports["PromptTemplate"]
    template = (
        "你是一名可靠的学习助手。请给出学习某个主题的合理入门顺序。\n\n"
        "【输入信息】\n"
        "- 对话历史：{history}\n"
        "- 用户问题：{question}\n"
        "- 当前主题：{subject}\n\n"
        "【回答规则】\n"
        "1. 用带序号的步骤给出入门顺序（先学什么、再学什么、最后做什么），每步一句话说明。\n"
        "2. 在步骤后补一个马上能动手的小练习，例如搭一个最小示例或解决一道典型问题。\n"
        "3. 结尾用一句话问用户是否需要一份详细的学习计划；不要替用户直接生成计划。\n"
        "4. 只围绕用户问的主题展开，不要混入软件测试、系统架构、混合检索、动态切分、Agent协作等无关内容。\n"
        "5. 不要编造具体出处；不确定的地方说明清楚。\n"
        "6. 不要写任何关于答案来源、知识库或模型本身的声明，直接给出学习顺序。\n"
        "7. 总字数控制在 150-300 字。\n"
    )
    return PromptTemplate.from_template(template)  # type: ignore


def build_general_prompt(imports: Dict[str, Any]):
    """纯通用知识提示词：无资料时使用，来源由系统内部字段记录。"""
    PromptTemplate = imports["PromptTemplate"]
    template = (
        "你是一名可靠的学习助手。当前知识库没有与问题相关的资料，请用你自身的通用知识回答。\n\n"
        "【输入信息】\n"
        "- 对话历史：{history}\n"
        "- 用户问题：{question}\n"
        "- 当前主题：{subject}\n\n"
        "【回答规则】\n"
        "1. 直接回答问题，给出准确、清晰的解释；可用要点列举。\n"
        "2. 若问题在询问某个具体资料/笔记/章节的内容，请回答「未找到你指定的资料，无法据此回答」。\n"
        "3. 不要编造具体出处、数字或年份；不确定的内容要说明不确定。\n"
        "4. 不要写任何关于答案来源、知识库或模型本身的声明，直接作答。\n"
        "5. 答案控制在 120-300 字。\n"
    )
    return PromptTemplate.from_template(template)  # type: ignore
