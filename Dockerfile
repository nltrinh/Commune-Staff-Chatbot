# Sử dụng image python slim để giảm dung lượng
FROM python:3.10-slim

# Cài đặt các thư viện hệ thống cần thiết cho xử lý file và build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy file requirements và cài đặt dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . .

# Tạo thư mục chứa dữ liệu upload và logs
RUN mkdir -p data/uploads logs

# Expose port 8000 cho FastAPI
EXPOSE 8000

# Lệnh khởi chạy ứng dụng (sẽ được override bởi docker-compose nếu cần)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
