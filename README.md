# MBC_final_project_EV_Predictive_Maintenance
전기차 충전기 예지보전 AI 모델(FastAPI 기반)

1. DATABASE 연결 방법 

.env 파일 생성 후 
DB 부분을 본인 DB정보 입력하기

DB_HOST=localhost
DB_PORT=5432
DB_NAME=myDB
DB_USER=postgres
DB_PASSWORD=1234

FASTAPI_BASE_URL=http://localhost:8003

-- 저장 되면 python파일은 .env파일에 있는 DB정보로 연결됩니다

2. 작동방법
파일 경로 fast_app으로 설정 ( cd fast* )
fastapi 실행 명령어 : uvicorn main:app --reload --port 8003
sensor_simulator.py : python sensor_simulator.py

시뮬레이터가 센서 데이터를 생성 할 때마다, 충전기 현황 및 그래프, 예지보전이 갱신
