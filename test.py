import pybullet as p
import os

# 1. 파일 경로 설정 (절대 경로로 변환 필수!)
# 현재 스크립트 위치 기준으로 상대 경로를 찾은 뒤 절대 경로로 바꿉니다.
#current_dir = os.path.dirname(os.path.abspath(__file__)) # 현재 파일 위치
input_stl = os.path.join(current_dir, "/src/assets/urdf/franka_panda/franka_panda_handle/meshes/visual/base_link.STL")
output_obj = os.path.join(current_dir, "/src/assets/urdf/franka_panda/franka_panda_handle/meshes/visual/base_link_vhacd.obj")
log_file = os.path.join(current_dir, "vhacd_log.txt")

print(f"Input: {input_stl}")
print(f"Output: {output_obj}")

# 2. 파일 존재 여부 확인
if not os.path.exists(input_stl):
    print("Error: 입력 파일을 찾을 수 없습니다! 경로를 확인하세요.")
else:
    print("Start V-HACD...")
    # 3. V-HACD 실행
    p.vhacd(input_stl, output_obj, log_file)
    print("Done!")