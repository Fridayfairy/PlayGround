# 原生 DB-API2.0 PyMySQL
# pip install pymysql
# bug: 安装pymysql时，需要安装cryptography，否则会报错:
# RuntimeError: 'cryptography' package is required for sha256_password or caching_sha2_password auth methods
# pip install cryptography 

import os
import pymysql
from contextlib import closing

DB_USER = os.getenv("DB_USER", "app")
DB_PASS = os.getenv("DB_PASS", "secret")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME", "demo")

print(f"DB_HOST={DB_HOST}, DB_PORT={DB_PORT}, DB_USER={DB_USER}, DB_PASS={DB_PASS}, DB_NAME={DB_NAME}")
print(f"type(DB_HOST)={type(DB_HOST)}, type(DB_PORT)={type(DB_PORT)}, type(DB_USER)={type(DB_USER)}, type(DB_PASS)={type(DB_PASS)}, type(DB_NAME)={type(DB_NAME)}")

def get_conn():
    return pymysql.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS,
        database=DB_NAME, 
        charset='utf8mb4', # 所有Unicode, 包含 emoji
        autocommit=False, # 关闭自动提交，需要手动提交事务
        cursorclass=pymysql.cursors.DictCursor # 游标返回字典格式，默认为元组
    )

def setup_schema():
    with closing(get_conn()) as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS `user` (
                `id` INT AUTO_INCREMENT PRIMARY KEY,
                `name` VARCHAR(255) NOT NULL,
                `email` VARCHAR(255) NOT NULL UNIQUE,
                `age` INT NOT NULL,
                `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB;
            """)
            conn.commit()

def seed_data():
    with closing(get_conn()) as conn, conn.cursor() as cursor:
        cursor.executemany("""
        INSERT INTO `user` (`name`, `email`, `age`)
        VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE name=VALUES(name), 
        age=VALUES(age);
        """, [
            ('Alice', 'alice@example.com', 25),
            ('Bob', 'bob@example.com', 21),
            ('Charlie', 'charlie@example.com', 22),
        ])
        conn.commit()

def query_examples():
    with closing(get_conn()) as conn, conn.cursor() as cursor:
        cursor.execute("SELECT id, name, email, age FROM `user` WHERE age >= %s",(22,))
        rows = cursor.fetchall()
        for row in rows:
            print(row)
        
        cursor.execute("SELECT COUNT(*) AS cnt FROM `user`")
        print("total num:", cursor.fetchone()["cnt"])

def update_and_txn_demo():
    with closing(get_conn()) as conn, conn.cursor() as cursor:
        try:
            cursor.execute("UPDATE `user` SET age=%s+1 WHERE name=%s", (23, "Alice"))
            cursor.execute("INSERT INTO `user`(id, name, email) VALUES (1, 'Dup', 'dup@example.com')") # 插入重复 id 报错
            conn.commit()
        except Exception as e:
            print("update failed:", e)
            conn.rollback()

def delete_demo():
    with closing(get_conn()) as conn, conn.cursor() as cursor:
        try:
            cursor.execute("DELETE FROM `user` WHERE age=%s", (23,))
            conn.commit()
        except Exception as e:
            print("delete failed:", e)
            conn.rollback()

if __name__ == '__main__':
    setup_schema()
    seed_data()
    query_examples()
    update_and_txn_demo()
    delete_demo()
    query_examples()
