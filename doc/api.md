
## api 文档

### 1. 全局接口定义

输入参数

| 参数      | 类型   | 说明                          | 示例        |
| --------- | ------ | ----------------------------- | ----------- |
| appId     | string | 应用渠道编号                  |             |
| version   | string | 版本号                        |             |
| signType  | string | 签名算法，目前使用国密SM2算法 | SM2或SHA256 |
| signData  | string | 签名数据，具体算法见下文      |             |
| encType   | string | 接口数据加密算法，目前不加密  | plain       |
| timestamp | int    | unix时间戳（秒）              |             |
| data      | json   | 接口数据，详见各接口定义      |             |

> 签名/验签算法：
>
> 1. 筛选，获取参数键值对，剔除signData、encData、extra三个参数。data参数按key升序排列进行json序列化。
> 2. 排序，按key升序排序。
> 3. 拼接，按排序好的顺序拼接请求参数
>
> ```key1=value1&key2=value2&...&key=appSecret```，key=appSecret固定拼接在参数串末尾，appSecret需替换成应用渠道所分配的appSecret。
>
> 4. 签名，使用制定的算法进行加签获取二进制字节，使用 16进制进行编码Hex.encode得到签名串，然后base64编码。
> 5. 验签，对收到的参数按1-4步骤签名，比对得到的签名串与提交的签名串是否一致。

签名示例：

```json
请求参数：
{
    "appId":"19E179E5DC29C05E65B90CDE57A1C7E5",
    "version": "1",
    "signType": "SM2",
    "signData": "...",
    "encType": "plain",
    "timestamp":1591943910,
    "data": {
    	"user_id":"gt",
    	"face_id":"5ed21b1c262daabe314048f5"
    }
}

密钥：
appSecret="D91CEB11EE62219CD91CEB11EE62219C"
SM2_privateKey="JShsBOJL0RgPAoPttEB1hgtPAvCikOl0V1oTOYL7k5U="

待加签串：
appId=19E179E5DC29C05E65B90CDE57A1C7E5&data={"face_id":"5ed21b1c262daabe314048f5","user_id":"gt"}&encType=plain&signType=SM2&timestamp=1591943910&version=1&key=D91CEB11EE62219CD91CEB11EE62219C

SHA256加签结果：
"2072bd8afb678c03ce9be14202e47b12031aa42a0a8c8593723d7027007ef804"

base64后结果：
"MjA3MmJkOGFmYjY3OGMwM2NlOWJlMTQyMDJlNDdiMTIwMzFhYTQyYTBhOGM4NTkzNzIzZDcwMjcwMDdlZjgwNA=="

SM2加签结果（每次不同）：
"LXgGBQNsXwofSXr+uXYiw0al7MFNNdUl0OyjpxHGKSPjJAr1N5oO6Tq3WL0C8UVX1pmDNH/GZK1Q0h+VvzKiEg=="


```

返回结果

| 参数      | 类型    | 说明                                                         | 示例  |
| --------- | ------- | ------------------------------------------------------------ | ----- |
| appId     | string  | 应用渠道编号                                                 |       |
| code      | string  | 接口返回状态代码                                             |       |
| signType  | string  | 签名算法，plain： 不用签名，SM2：使用SM2算法                 | plain |
| encType   | string  | 接口数据加密算法，目前不加密                                 | plain |
| success   | boolean | 成功与否                                                     |       |
| timestamp | int     | unix时间戳                                                   |       |
| data      | json    | 成功时返回结果数据；出错时，data.msg返回错误说明。详见具体接口 |       |

> 成功时：code为0， success为True，data内容见各接口定义；
>
> 出错时：code返回错误代码，具体定义见各接口说明

返回示例

```json
{
    "appId": "19E179E5DC29C05E65B90CDE57A1C7E5", 
    "code": 0, 
    "signType": "plain",
    "encType": "plain",
    "success": true,
    "timestamp": 1591943910,
    "data": {
       "msg": "success", 
       ...
    }
}
```

全局出错代码

| 编码 | 说明                               |
| ---- | ---------------------------------- |
| 9800 | 无效签名                           |
| 9801 | 签名参数有错误                     |
| 9802 | 调用时间错误，unixtime超出接受范围 |



### 2. 抗原检测结果识别

> 识别抗原检测试剂盒正面的结果。
>
> 注意：
>
> 1. 照片中只能出现一个试剂盒正面，出现多个会影响结果
> 2. 试剂盒需呈横向或纵向摆放，允许小角度倾斜，过渡倾斜可能会影响识别结果
> 3. 尽量不要大面积遮挡试剂盒正面

请求URL

> http://127.0.0.1:5000/antigen/check

请求方式

> POST

输入参数

| 参数  | 必选 | 类型   | 说明               |
| ----- | ---- | ------ | ------------------ |
| image | 是   | string | base64编码图片数据 |

请求示例

```json
{
    "image" : "...."
}
```

返回结果

| 参数       | 必选 | 类型   | 说明                 |
| ---------- | ---- | ------ | -------------------- |
| result     | 是   | string | 粗分识别结果，见下表 |
| comment    | 是   | string | 细分识别结果，见下表 |
| request_id | 是   | string | 此次请求id           |

> 识别结果说明：
>
> | result   | comment | 识别结果                     |
> | -------- | ------- | ---------------------------- |
> | positive | pos     | 阳性：C、T 均有标线          |
> | negative | neg     | 阴性：仅 C 有标线            |
> | invalid  | fal     | 无效：仅 T 有标线            |
> | invalid  | nul     | 无效：C、T 均无标线          |
> | invalid  | none    | 无效：未定位到 C、T 标线区域 |
>

返回示例

```json
{
    "appId": "66A095861BAE55F8735199DBC45D3E8E", 
    "code": 0, 
    "success": true, 
    "signType": "plain", 
    "encType": "plain", 
    "data": {
        "msg": "success", 
        "result": "negative", 
        "comment": "neg",
        "request_id": "22041814265675a198ffdf7eaafecb7e9724a56cdfd9"
    }, 
    "timestamp": 1650263216
}

```

出错代码

| 编码 | 说明                              |
| ---- | --------------------------------- |
| 9001 | 缺少参数                          |
| 9002 | 图片数据太大，base64数据不大于3MB |
| 9901 | base64编码异常                    |

