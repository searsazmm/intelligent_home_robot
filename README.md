# intelligent_home_robot

Django 5.2 项目（Python 3.10）。

## 运行

```bash
cd C:\projects\intelligent_home_robot
python manage.py migrate      # 初始化数据库（默认 SQLite）
python manage.py runserver    # 启动开发服务器 http://127.0.0.1:8000
```

## 结构

```
intelligent_home_robot/
├── manage.py
├── intelligent_home_robot/   # 项目配置包
│   ├── settings.py           # 配置（数据库、应用注册等）
│   ├── urls.py               # 根路由
│   ├── asgi.py / wsgi.py     # 部署入口
│   └── __init__.py
```

## 常用命令

```bash
python manage.py startapp <app名>   # 新建应用
python manage.py makemigrations     # 生成迁移
python manage.py createsuperuser    # 建管理员（后台 /admin）
```

> 当前使用全局 Python 环境（已装 Django 5.2.14）。如需环境隔离，可执行 `python -m venv venv` 后用 `venv\Scripts\pip install django` 重建。
