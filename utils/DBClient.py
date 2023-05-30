import mariadb
import logging
from dotenv import dotenv_values
from typing import Optional


class DBConn:
    conn = None

    def __init__(self):
        env: dict = dotenv_values()
        self.conn = mariadb.connect(
            user=env.get("DB_USER"),
            password=env.get("DB_PASS"),
            host=env.get("DB_HOST"),
            port=int(env.get("DB_PORT")),
            database=env.get("DB_DATABASE")
        )
        logging.info("DB Client is up")

    def get_non_detected(self, limit: int = 16):
        cur = self.conn.cursor()
        cur.execute("SELECT"
                    "  l.id as id,"
                    "  i.url as url"
                    " FROM"
                    "  device_log AS l"
                    "   INNER JOIN image AS i"
                    "    on l.image_id = i.id"
                    "   INNER JOIN detect as d"
                    "    on l.id = d.id"
                    " WHERE"
                    "  d.status = 'not_started'"
                    " ORDER BY l.date_created asc"
                    " LIMIT ?", (limit,))
        if cur is None:
            return []
        r = [(log_id, url) for log_id, url in cur]
        logging.debug(f"Got db select: {r}")
        cur.close()
        return r

    def update_detected(self, log_id: int, status='done', result: Optional[str] = None):
        cur = self.conn.cursor()
        try:
            cur.execute("UPDATE detect"
                        " SET"
                        "  status = ?,"
                        "  result = ?"
                        " WHERE id = ?", (status, result, log_id,))
        except mariadb.Error as e:
            logging.error(f"Maria DB Error: {e}")
        finally:
            r = cur.lastrowid
            logging.debug(f"Got db update: {r}")
            cur.close()
        self.conn.commit()
        return r

    def __del__(self):
        self.conn.close()
