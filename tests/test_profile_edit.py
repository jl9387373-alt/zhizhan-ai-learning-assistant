import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app as app_module
from storage.profiles import ProfilesStore


class TestProfileEditApi(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original_store_factory = app_module.ProfilesStore
        self.original_avatar_dir = app_module.AVATAR_DIR
        app_module.ProfilesStore = lambda: ProfilesStore(self.tmp.name)
        app_module.AVATAR_DIR = os.path.join(self.tmp.name, 'avatars')
        self.client = app_module.app.test_client()

    def tearDown(self):
        app_module.ProfilesStore = self.original_store_factory
        app_module.AVATAR_DIR = self.original_avatar_dir
        self.tmp.cleanup()

    def get_profile(self):
        response = self.client.get('/api/auth/profile?user_id=u1')
        self.assertEqual(response.status_code, 200)
        return response.get_json()['data']

    def test_update_nickname_and_avatar(self):
        response = self.client.post('/api/auth/profile', json={
            'user_id': 'u1',
            'nickname': '小学习者',
            'avatar': 'rocket',
        })
        self.assertEqual(response.status_code, 200)

        profile = self.get_profile()
        self.assertEqual(profile['nickname'], '小学习者')
        self.assertEqual(profile['avatar'], 'rocket')
        self.assertEqual(profile['level'], 1)

    def test_reject_invalid_values(self):
        self.client.post('/api/auth/profile', json={
            'user_id': 'u1',
            'nickname': '原昵称',
            'avatar': 'default',
        })

        invalid_avatar = self.client.post('/api/auth/profile', json={
            'user_id': 'u1',
            'avatar': 'unknown',
        })
        self.assertEqual(invalid_avatar.status_code, 400)

        empty_nickname = self.client.post('/api/auth/profile', json={
            'user_id': 'u1',
            'nickname': '   ',
        })
        self.assertEqual(empty_nickname.status_code, 400)

        profile = self.get_profile()
        self.assertEqual(profile['nickname'], '原昵称')
        self.assertEqual(profile['avatar'], 'default')

    def test_upload_custom_avatar(self):
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"fake-png-content"
        response = self.client.post('/api/profile/avatar', data={
            'user_id': 'u1',
            'nickname': '带头像',
            'file': (io.BytesIO(png_bytes), 'avatar.png'),
        }, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)

        profile = self.get_profile()
        self.assertEqual(profile['nickname'], '带头像')
        self.assertEqual(profile['avatar'], 'custom')
        self.assertTrue(profile.get('avatar_version'))

        avatar_response = self.client.get('/api/avatar?user_id=u1', buffered=True)
        self.assertEqual(avatar_response.status_code, 200)
        self.assertEqual(avatar_response.data, png_bytes)

    def test_reject_non_image_avatar(self):
        response = self.client.post('/api/profile/avatar', data={
            'user_id': 'u1',
            'file': (io.BytesIO(b'not an image'), 'avatar.txt'),
        }, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)

        profile = self.get_profile()
        self.assertEqual(profile['avatar'], 'default')


if __name__ == '__main__':
    unittest.main()
