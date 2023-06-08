import mariadb
import logging
import os
from typing import Optional


class DBConn:
    conn = None

    def __init__(self):
        self.conn = mariadb.connect(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT")),
            database=os.getenv("DB_DATABASE")
        )
        self.limit = int(os.getenv("DB_LIMIT", "16"))
        logging.info("DB Client is up")

    def get_non_detected(self, limit: Optional[int] = None):
        limit = self.limit if limit is None else limit
        cur = self.conn.cursor()
        cur.execute("SELECT"
                    "  l.id AS id,"
                    "  i.url AS url"
                    " FROM"
                    "  device_log AS l"
                    "   INNER JOIN image AS i"
                    "    ON l.image_id = i.id"
                    "   INNER JOIN detect AS d"
                    "    ON l.id = d.id"
                    " WHERE"
                    "  d.status = 'not_started'"
                    " ORDER BY l.date_created ASC"
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
