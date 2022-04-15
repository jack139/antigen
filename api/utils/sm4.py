# -*- coding: utf-8 -*-

# SM2 验签： 与 yhtool-crypto-1.3.0-RELEASE.jar yhtool-sdk-1.3.0-RELEASE.jar 测试通过

import base64
import binascii
from gmssl.sm4 import CryptSM4, SM4_ENCRYPT, SM4_DECRYPT

from config.settings import SECRET_KEY

crypt_sm4 = CryptSM4()

sm4_keysize = 16;
sm4_transformation = "SM4/ECB/PKCS7Padding";

# 加密
def encrypt(data, key):
    crypt_sm4.set_key(key, SM4_ENCRYPT)
    return crypt_sm4.crypt_ecb(data)

# 解密
def decrypt(data, key):
    crypt_sm4.set_key(key, SM4_DECRYPT)
    return crypt_sm4.crypt_ecb(data)

# 生成 key
def generateEncKey(appId, appSecret):
    firstKey = appId[:sm4_keysize]
    appSecretEncData = encrypt(appSecret, firstKey);
    return binascii.b2a_hex(appSecretEncData)[:sm4_keysize]

# 加密数据
def encrypt_data(appid, plain_data_bytes):
    if appid not in SECRET_KEY.keys():
        return None
    encKey = generateEncKey(appid.encode('utf-8'), SECRET_KEY[appid].encode('utf-8'));
    encData = encrypt(plain_data_bytes, encKey)
    return base64.b64encode(encData)

# 解密数据
def decrypt_data(appid, encrypt_data_base64):
    if appid not in SECRET_KEY.keys():
        return None
    encKey = generateEncKey(appid.encode('utf-8'), SECRET_KEY[appid].encode('utf-8'));
    encrypt_data_bytes = base64.b64decode(encrypt_data_base64)
    return decrypt(encrypt_data_bytes, encKey)


if __name__ == '__main__':
    data = b"321"
    encData = encrypt_data('19E179E5DC29C05E65B90CDE57A1C7E5', b'321')
    print(encData)
    data2 = decrypt_data('19E179E5DC29C05E65B90CDE57A1C7E5', encData)
    print(data2)
