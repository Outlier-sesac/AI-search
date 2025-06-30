import os
import pyodbc
import requests  # Bing Search 테스트를 위해 추가
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential

# .env 파일에서 모든 환경 변수 로드
load_dotenv()

print("="*60)
print("     Azure 서비스 연결 통합 테스트 스크립트")
print("="*60)

# ===================================================
# 1. Azure AI Search 연결 테스트
# ===================================================
print("\n--- 1. Azure AI Search 연결 테스트 시작 ---")

# 연결 설정 정보 로드
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")

try:
    if not all([search_endpoint, search_key]):
        raise ValueError("AZURE_SEARCH_ENDPOINT 또는 AZURE_SEARCH_ADMIN_KEY 환경 변수가 설정되지 않았습니다.")
    
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

sql_connection = None  # finally 블록에서 사용하기 위해 미리 선언

try:
    if not all([sql_server, sql_database, sql_username, sql_password]):
        raise ValueError("Azure SQL 데이터베이스 연결 정보(서버, DB, 사용자, 암호)가 .env 파일에 모두 설정되지 않았습니다.")

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

    print("🔄 데이터베이스에 연결을 시도합니다...")
    sql_connection = pyodbc.connect(connection_string)
    print("✅ 데이터베이스 연결 성공!")

    cursor = sql_connection.cursor()
    cursor.execute("SELECT @@VERSION")
    row = cursor.fetchone()
    if row:
        print(f"   - 서버 버전: {row[0][:30]}...")

except Exception as ex:
    print(f"❌ 데이터베이스 연결 실패!")
    print(f"   - 에러: {ex}")
    print("\n   [체크리스트]")
    print("   1. .env 파일의 SQL 관련 정보가 정확한가요?")
    print("   2. Azure Portal에서 VM의 IP가 SQL 방화벽 규칙에 추가되었나요?")
    print("   3. VM에 ODBC Driver 18이 설치되었나요?")

finally:
    if sql_connection:
        sql_connection.close()
        print("\n   🚪 데이터베이스 연결을 닫았습니다.")

