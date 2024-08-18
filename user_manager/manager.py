import os
import logging

from flask import Flask, render_template, request, redirect, url_for, flash
import psycopg2
from logfmter import Logfmter

app = Flask(__name__, template_folder='templates')
app.secret_key = os.getenv("FLASK_SECRET_KEY")

# Enable logging
formatter = Logfmter(
    keys=["at", "logger", "level", "msg"],
    mapping={"at": "asctime", "logger": "name", "level": "levelname", "msg": "message"},
    datefmt='%H:%M:%S %d/%m/%Y'
)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[stream_handler]
)

# Set a higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def get_db_connection():
    try:
        logger.info("Establishing database connection.")
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD")
        )
        return conn
    except Exception as e:
        logger.error(f"Error establishing database connection: {e}")
        raise


@app.route('/')
def index():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT au.user_id, ub.balance, ub.images_generated AS images_generated
            FROM allowed_users au
            LEFT JOIN user_balances ub ON au.user_id = ub.user_id
            ORDER BY au.user_id
        """)
        allowed_users = cur.fetchall()
        logger.info("Fetched allowed users successfully.")
        return render_template('index.html', allowed_users=allowed_users)
    except Exception as e:
        logger.error(f"Error fetching allowed users: {e}")
        flash("Could not load allowed users.", 'error')
        return render_template('index.html', allowed_users=[])
    finally:
        logger.info("Closing database connection.")
        cur.close()
        conn.close()


@app.route('/allow', methods=['POST'])
def allow_user():
    user_id = request.form.get('user_id')
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT user_id FROM allowed_users WHERE user_id = %s", (user_id,))
        existing_user = cur.fetchone()
        if existing_user:
            flash(f"User {user_id} is already allowed.", 'info')
            logger.info(f"User {user_id} is already allowed.")
        else:
            cur.execute("INSERT INTO allowed_users (user_id) VALUES (%s)", (user_id,))
            conn.commit()
            flash(f"User {user_id} has been allowed.", 'success')
            logger.info(f"User {user_id} has been successfully allowed.")
    except Exception as e:
        logger.error(f"Error allowing user {user_id}: {e}")
        flash(f"Error allowing user {user_id}.", 'error')
    finally:
        logger.info("Closing database connection.")
        cur.close()
        conn.close()
    return redirect(url_for('index'))


@app.route('/disable', methods=['POST'])
def disable_user():
    user_id = request.form.get('user_id')
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT user_id FROM allowed_users WHERE user_id = %s", (user_id,))
        existing_user = cur.fetchone()
        if not existing_user:
            flash(f"User {user_id} is not currently allowed.", 'info')
            logger.info(f"User {user_id} is not currently allowed.")
        else:
            cur.execute("DELETE FROM allowed_users WHERE user_id = %s", (user_id,))
            conn.commit()
            flash(f"User {user_id} access revoked.", 'warning')
            logger.info(f"User {user_id} access revoked.")
    except Exception as e:
        logger.error(f"Error disabling user {user_id}: {e}")
        flash(f"Error disabling user {user_id}.", 'error')
    finally:
        logger.info("Closing database connection.")
        cur.close()
        conn.close()
    return redirect(url_for('index'))


@app.route('/set_balance', methods=['POST'])
def set_balance():
    user_id = request.form.get('user_id')
    balance = request.form.get('balance')
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT user_id FROM allowed_users WHERE user_id = %s", (user_id,))
        existing_user = cur.fetchone()
        if not existing_user:
            flash(f"User {user_id} is not currently allowed.", 'info')
            logger.info(f"User {user_id} is not currently allowed when setting balance.")
        else:
            cur.execute(
                """
                INSERT INTO user_balances (user_id, balance, images_generated)
                VALUES (%s, %s, 0)
                ON CONFLICT (user_id)
                DO UPDATE SET balance = EXCLUDED.balance
                """,
                (user_id, balance)
            )
            conn.commit()
            flash(f"User {user_id} balance has been set to {balance}.", 'success')
            logger.info(f"User {user_id} balance set to {balance}.")
    except Exception as e:
        logger.error(f"Error setting balance for user {user_id}: {e}")
        flash(f"Error setting balance for user {user_id}.", 'error')
    finally:
        logger.info("Closing database connection.")
        cur.close()
        conn.close()
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
