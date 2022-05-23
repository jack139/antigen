
### CentOS 7 需要安装的包

```shell
yum update
yum groupinstall "Development Tools"

yum install epel-release
yum install python36
yum install python3-devel openssl-devel pcre-devel
yum install nginx
yum install cmake

pip3 install -U pip
pip3 install uwsgi
```



### python3 需要安装的包
见 requirement.txt



### nginx配置文件

> nginx.conf:

```nginx
client_max_body_size 8m;

server {
    listen       5000;

    access_log   /var/log/nginx/access_antigen.log;

    location / {
        include uwsgi_params;
        uwsgi_pass unix:/tmp/uwsgi_antigen.sock;
        uwsgi_param UWSGI_CHDIR /usr/share/nginx/html/antigen;
        uwsgi_param UWSGI_SCRIPT app:app;
    }

    location /static/ {
        root /usr/share/nginx/html/antigen;
    }

    error_page 500 502 503 504 /50x.html;
    location = /50x.html {
    }
}
```



### Nginx关闭自动压缩日志
```
rm /etc/logrotate.d/nginx
```



## redis相关设置
```
sysctl vm.overcommit_memory=1
sysctl net.core.somaxconn=1000
echo never > /sys/kernel/mm/transparent_hugepage/enabled
```

前两行可在/etc/sysctl.conf中增加
```
vm.overcommit_memory=1
net.core.somaxconn=1000
```
然后执行 sysctl -p

第三行放入/etc/rc.local



### 编译 sm3
```
cd src/api/utils/libsm3/
gcc -fPIC -shared -o libsm3.so sm3.c
```



### 相关shell脚本

shell/redis_server  启停redis
shell/my_server 启停应用服务



### 定时重启

在 crontab -e 设置
```
0 5 * * * /usr/share/nginx/html/log_cut.sh > /tmp/log_cut.log
30 3 * * * /root/my_server restart > /tmp/restart_back.log 2>&1
```

新建uwsgi文件链接，防止crond找不到
```
ln -s /usr/local/bin/uwsgi /usr/bin/
ln -s /usr/sbin/nginx /usr/bin/
```
