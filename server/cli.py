import argparse
from dataclasses import fields, replace
import getpass
import json
import logging
import os
from pathlib import Path
import re
import tempfile

from server.config import Config
from server.store import Store


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="GuitarOCR 多用户服务")
    parser.add_argument(
        "command", choices=("init", "serve", "api", "worker", "admin", "validate")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("output/server/config.json")
    )
    parser.add_argument("--data", type=Path, default=Path("output/server"))
    parser.add_argument("--public-url", default="http://localhost:8080")
    parser.add_argument(
        "--gpus", default="0", help="GPU 编号，以逗号分隔；空字符串关闭 GPU"
    )
    parser.add_argument("--username", default="admin")
    parser.add_argument("--engine", choices=("gpu", "browser"), default="gpu")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    if args.command == "init":
        config = Config(args.data, public_url=args.public_url, gpus=args.gpus)
        if (config.data / "config.json").exists():
            parser.error("配置已存在，修改配置文件或使用 admin 子命令")
        if not re.fullmatch(r"[A-Za-z0-9_-]{3,32}", args.username):
            parser.error("用户名使用 3 至 32 个字母、数字、下划线或连字符")
        password = os.environ.pop("GUITAROCR_ADMIN_PASSWORD", None) or getpass.getpass(
            "管理员密码（至少 10 个字符）: "
        )
        Store(config.database).add_user(args.username, password, admin=True)
        print(f"已创建管理员 {args.username}，配置保存在 {config.save()}")
        return
    config = Config.load(args.config)
    if args.command == "validate":
        from server.app import create_app

        with tempfile.TemporaryDirectory() as directory:
            create_app(replace(config, data=Path(directory)))
        for path in (
            Path(config.model) / "config.json",
            Path(config.layout_model) / "inference.pdiparams",
            Path(config.info_adapter) / "adapter_model.safetensors",
            Path(config.measure_adapter) / "adapter_model.safetensors",
        ):
            if not path.is_file() or path.stat().st_size < 256:
                raise ValueError(f"模型文件缺失：{path}")
        from shared.environment import verify_files

        manifest = json.loads(
            (Path(config.source) / "weights/manifest.json").read_text()
        )
        errors = [
            error
            for model in manifest["models"]
            for error in verify_files(
                Path(config.source) / model["path"], model["files"]
            )
        ]
        if errors:
            raise ValueError("模型校验失败：" + "; ".join(errors))
        print(json.dumps({"ok": True, "config_fields": len(fields(config))}))
    elif args.command == "admin":
        store = Store(config.database)
        if not re.fullmatch(r"[A-Za-z0-9_-]{3,32}", args.username):
            parser.error("用户名格式有误")
        password = getpass.getpass("新管理员密码: ")
        from server.store import password_hash

        encoded = password_hash(password)
        user = store.one("SELECT * FROM users WHERE username=?", (args.username,))
        if user:
            with store.connect(True) as db:
                db.execute(
                    "UPDATE users SET password=?,admin=1,disabled=0 WHERE id=?",
                    (encoded, user["id"]),
                )
                db.execute("DELETE FROM sessions WHERE user_id=?", (user["id"],))
        else:
            store.add_user(args.username, password, admin=True)
        print("管理员账号已保存")
    elif args.command == "serve":
        from server.supervisor import Supervisor
        import fcntl

        with (config.data / "supervisor.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error("此数据目录已有服务运行")
            Supervisor(config, args.config).run()
    elif args.command == "api":
        import uvicorn
        from server.app import create_app

        uvicorn.run(
            create_app(config),
            host=config.host,
            port=config.port,
            proxy_headers=True,
            forwarded_allow_ips="127.0.0.1,::1",
            workers=1,
        )
    else:
        if args.engine == "gpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        from server.worker import run

        run(config, args.engine, "cuda:0" if args.engine == "gpu" else "cpu")


if __name__ == "__main__":
    main()
