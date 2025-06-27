import os
import pyodbc
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential

# .env 파일에서 모든 환경 변수 로드
load_dotenv()

# ===================================================
# 1. Azure AI Search 연결 테스트
# ===================================================
print("--- 1. Azure AI Search 연결 테스트 시작 ---")

# 연결 설정 정보 로드
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")

try:
    if not all([search_endpoint, search_key]):
        raise ValueError("AZURE_SEARCH_ENDPOINT 또는 AZURE_SEARCH_ADMIN_KEY 환경 변수가 설정되지 않았습니다.")
    
    # 자격 증명 객체 생성 (이것만으로도 키 형식의 유효성을 일부 확인 가능)
    credential = AzureKeyCredential(search_key)
    
    print(f"✅ Azure AI Search 연결 설정 확인 완료!")
    print(f"   - 엔드포인트: {search_endpoint}")

except Exception as e:
    print(f"❌ Azure AI Search 연결 설정 실패: {e}")


# ===================================================
# 2. Azure SQL Database 연결 테스트
# ===================================================
print("\n--- 2. Azure SQL Database 연결 테스트 시작 ---")

# 연결 설정 정보 로드
sql_server = os.getenv("AZURE_SQL_SERVER")
sql_database = os.getenv("AZURE_SQL_DATABASE")
sql_username = os.getenv("AZURE_SQL_USER")
sql_password = os.getenv("AZURE_SQL_PASSWORD")

connection = None  # finally 블록에서 사용하기 위해 미리 선언

try:
    if not all([sql_server, sql_database, sql_username, sql_password]):
        raise ValueError("Azure SQL 데이터베이스 연결 정보(서버, DB, 사용자, 암호)가 .env 파일에 모두 설정되지 않았습니다.")

    # pyodbc를 위한 연결 문자열 생성
    # 드라이버 이름은 시스템에 설치된 것과 일치해야 합니다 (ODBC Driver 18 for SQL Server)
    connection_string = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER=tcp:{sql_server},1433;"
        f"DATABASE={sql_database};"
        f"UID={sql_username};"
        f"PWD={sql_password};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
        f"Connection Timeout=30;"
    )

    # 데이터베이스에 연결
    print("🔄 데이터베이스에 연결을 시도합니다...")
    connection = pyodbc.connect(connection_string)
    print("✅ 데이터베이스 연결 성공!")

    # 간단한 쿼리로 실제 통신 확인
    cursor = connection.cursor()
    cursor.execute("SELECT @@VERSION")
    row = cursor.fetchone()
    if row:
        print(f"   - 서버 버전: {row[0][:30]}...") # 너무 길어서 앞부분만 출력

except Exception as ex:
    print(f"❌ 데이터베이스 연결 실패!")
    print(f"   - 에러: {ex}")
    print("\n   [체크리스트]")
    print("   1. .env 파일의 서버, DB, 사용자, 암호 정보가 정확한가요?")
    print("   2. Azure Portal에서 이 VM의 IP를 방화벽 규칙에 추가했나요?")
    print("   3. VM에 ODBC Driver 18이 올바르게 설치되었나요?")

finally:
    # 연결이 성공적으로 생성되었다면 반드시 닫아줍니다.
    if connection:
        connection.close()
        print("\n🚪 데이터베이스 연결을 닫았습니다.")