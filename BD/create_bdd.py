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

        # Create tables
        tables = [
            """
            CREATE TABLE IF NOT EXISTS Customers (
                customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                Gender_Female INT,
                Gender_Male INT,
                Customer_Type_Loyal_Customer INT,
                Customer_Type_Disloyal_Customer INT,
                Age INT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS Flights (
                flight_id INTEGER PRIMARY KEY AUTOINCREMENT,
                Flight_Distance FLOAT,
                Class_Business INT,
                Class_Eco INT,
                Class_Eco_Plus INT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS Travel (
                travel_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INT,
                Type_of_Travel_Business_Travel INT,
                Type_of_Travel_Personal_Travel INT,
                Departure_Delay_in_Minutes INT,
                Arrival_Delay_in_Minutes INT,
                FOREIGN KEY (customer_id) REFERENCES Customers(customer_id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS Services (
                service_id INTEGER PRIMARY KEY AUTOINCREMENT,
                travel_id INT,
                Inflight_wifi_service INT,
                Departure_Arrival_time_convenient INT,
                Ease_of_Online_booking INT,
                Gate_location INT,
                Food_and_drink INT,
                Online_boarding INT,
                Seat_comfort INT,
                Inflight_entertainment INT,
                On_board_service INT,
                Leg_room_service INT,
                Baggage_handling INT,
                Checkin_service INT,
                Inflight_service INT,
                Cleanliness INT,
                FOREIGN KEY (travel_id) REFERENCES Travel(travel_id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS Satisfaction (
                satisfaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                travel_id INT,
                Satisfaction TEXT,
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