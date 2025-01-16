from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, ForeignKey
import sqlite3

def create_connection(db_file):
    """Creer conexion a la base donnees SQLite."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"Conectea la base de donees: {db_file}")
    except sqlite3.Error as e:
        print(f"Error dans la conexion: {e}")
    return conn

if __name__ == "__main__":
    database = "./BD/plane_satisfaction.db"
    conn = create_connection(database)
    if conn:
        conn.close()
	# Close the cursor

def create_table(conn):
    try:
        cursor = conn.cursor()

        tables = [
            """
            CREATE TABLE IF NOT EXISTS Customers (
                customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                gender TEXT,
                customer_type TEXT,
                age INT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS Flights (
                flight_id INTEGER PRIMARY KEY AUTOINCREMENT,
                flight_distance FLOAT,
                class TEXT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS Travel (
                travel_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INT,
                type_of_travel TEXT,
                departure_delay INT,
                arrival_delay INT,
                FOREIGN KEY (customer_id) REFERENCES Customers(customer_id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS Services (
                service_id INTEGER PRIMARY KEY AUTOINCREMENT,
                travel_id INT,
                inflight_wifi_service INT,
                departure_arrival_time_convenient INT,
                ease_of_online_booking INT,
                gate_location INT,
                food_and_drink INT,
                online_boarding INT,
                seat_comfort INT,
                inflight_entertainment INT,
                on_board_service INT,
                leg_room_service INT,
                baggage_handling INT,
                checkin_service INT,
                inflight_service INT,
                cleanliness INT,
                FOREIGN KEY (travel_id) REFERENCES Travel(travel_id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS Satisfaction (
                satisfaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                travel_id INT,
                satisfaction TEXT,
                FOREIGN KEY (travel_id) REFERENCES Travel(travel_id)
            );
            """
        ]

        for table in tables:
            cursor.execute(table)

        print("Tables creer avec succes.")
        cursor.close()
    except sqlite3.Error as e:
        print(f"Erreur dans la creation de table: {e}")

def main():
    """ function pour creer la base de donnees """

    database = "./BD/plane_satisfaction.db"
    
    # Creer conexion 
    conn = create_connection(database)
    
    # Creer table
    if conn:
        create_table(conn)
        conn.close()

if __name__ == "__main__":
    main()