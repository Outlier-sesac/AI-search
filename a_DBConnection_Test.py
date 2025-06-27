import os
import pyodbc
from dotenv import load_dotenv

load_dotenv()

server = os.getenv("AZURE_SQL_SERVER")
database = os.getenv("AZURE_SQL_DATABASE")
username = os.getenv("AZURE_SQL_USER")
password = os.getenv("AZURE_SQL_PASSWORD")

table_name = input("조회할 테이블 이름을 입력하세요 (예: dbo.merged_A): ")

connection = None
try:
    if not all([server, database, username, password]):
        raise ValueError("Azure SQL 데이터베이스 연결 정보가 .env 파일에 모두 설정되지 않았습니다.")

    connection_string = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER=tcp:{server},1433;"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        f"Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
    )

    connection = pyodbc.connect(connection_string)
    cursor = connection.cursor()

    query = f"SELECT TOP 10 * FROM {table_name}"
    print(f"\n🔄 '{table_name}' 테이블에서 상위 10개 데이터를 조회합니다...")
    cursor.execute(query)

    rows = cursor.fetchall()
    
    if not rows:
        print(f"-> 결과: '{table_name}' 테이블에 데이터가 없습니다.")
    else:
        # 컬럼 이름 출력
        column_names = [column[0] for column in cursor.description]
        print(" | ".join(column_names))
        print("-" * (len(" | ".join(column_names)) + 10))

        # 데이터 행 출력
        for row in rows:
            print(" | ".join(str(value) for value in row))
            
    print("\n✅ 조회 성공!")

except pyodbc.Error as ex:
    print(f"❌ 데이터베이스 오류 발생: {ex}")
    print("\n[체크리스트]")
    print("1. 입력한 테이블 이름이 정확한가요? (예: dbo.my_table)")
    print("2. 해당 테이블에 접근할 수 있는 권한이 있나요?")

finally:
    if connection:
        connection.close()
        print("\n🚪 데이터베이스 연결을 닫았습니다.")