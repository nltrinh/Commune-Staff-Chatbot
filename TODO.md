# KẾ HOẠCH DỰ ÁN

## Đang làm (IN PROGRESS)
## Đã xong (DONE)
- [x] Dọn dẹp `config.py` (bỏ hardcode, ưu tiên `.env`, thêm cờ `USE_MOCK_AI`).
- [x] Sửa `pipeline.py` để hỗ trợ trả về Mock Data khi test ở Local.
- [x] Test sơ bộ luồng Đăng nhập -> Chat (Mock AI).
- [x] Tạo một tài khoản host_admin để quản lý các tài khoản nhân viên và các thông tin khác.
- [x] Xóa các phần liên quan đến đăng ký đi chỉ cần host_admin cấp tài khoản nhân sự.
- [x] Thêm tính năng thay đổi mật khẩu cho Cán bộ (Backend + Frontend Modal).
- [x] Bảng quản lý nhân sự Admin: Thêm cột mật khẩu (ẩn/hiện bằng eye icon) nằm sau cột Tên đăng nhập.
- [x] Sửa thao tác xóa nhân sự (kết nối thực tế với Backend).
- [x] Thêm chức năng sửa thông tin nhân sự (Họ tên, Tuổi, Phòng ban, Mật khẩu).
- [x] Lưu lịch sử chat chi tiết và phân quyền xem lịch sử (Admin xem toàn bộ, User xem của mình).
- [x] Thống kê hoạt động của từng nhân sự (Số lượng câu hỏi).
- [x] Làm lại giao diện đẹp hơn xíu nữa, các nút nhấn trông hấp dẫn hơn, màu sắc hài hòa, chuyển động mượt mà. Có phần cài đặt sáng/tối. Trang đăng nhập nhìn bắt mắt hơn, có chuyện động kiểu vibe code.
- [x] Tạo module Quản lý phòng ban (CRUD phòng ban, liên kết nhân sự).
- [x] Chỉnh sửa quy trình xóa nhân sự: Điều hướng sang trang chi tiết phòng ban để xác nhận xóa tránh xóa nhầm.
- [x] Tự động tải danh sách phòng ban từ Database vào các select box (Form tạo tài khoản, modal nạp tài liệu).

## Sắp tới (TODO)
- [x] Bỏ nút sáng/tối phía bên dưới đi. Bỏ nút đổi mật khẩu đi. Bỏ nút đăng xuất phía bên dưới đi. Chuyển tên + chức vụ + avatar sang bên góc trên cùng bên phải, đổi hình avatar lại thành hình tròn, phía bên trái nó là nút chỉnh sáng/tối. Khi bấm vào avatar sẽ xổ xuống các tùy chọn: Hồ sơ, Đăng xuất. Khi bấm vào hồ sơ sẽ đến trang hồ sơ và trong đó có các thông tin người dùng + nút đổi mật khẩu. Khi bấm vào đăng xuất sẽ đăng xuất.

- [x] "Quản lý tài liệu" có thêm cột ai đã nạp tài liệu lên nữa.
## Đề xuất từ Agent (PROPOSALS)
- Tích hợp thêm các mẫu văn bản hành chính (Template) để nhân sự tải về nhanh.
