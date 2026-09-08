# GCCM-BE API 服务镜像
FROM python:3.11-slim

WORKDIR /app

# 系统依赖（scipy 编译所需）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY gccm_be ./gccm_be

RUN pip install --no-cache-dir numpy scipy

# 可选：鲁棒 MPC 后端（CasADi）
# RUN pip install --no-cache-dir casadi

EXPOSE 8080

# 启动配置化 API 服务（config 文件可挂载覆盖）
CMD ["python", "-m", "gccm_be.app.api", "--config", "/app/config.json"]
