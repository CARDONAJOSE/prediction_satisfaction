import sqlite3
import pandas as pd

# se connecter a la base donnees
data = pd.read_csv("./data/clean_data.csv")

# Se connecter à la base de données SQLite
conn = sqlite3.connect('./BD/plane_satisfaction.db')
cursor = conn.cursor()

# function de insertion table customer
def insert_customer(conn, gender_female, gender_male, customer_type_loyal, customer_type_disloyal, age):
    """Inserer un client dans la table Customers."""
    sql = '''INSERT INTO Customers(Gender_Female, Gender_Male, Customer_Type_Loyal_Customer, Customer_Type_Disloyal_Customer, Age)
             VALUES(?, ?, ?, ?, ?)'''
    cur = conn.cursor()
    cur.execute(sql, (gender_female, gender_male, customer_type_loyal, customer_type_disloyal, age))
    conn.commit()
    return cur.lastrowid

# function de insertion table flight
def insert_flight(conn, flight_distance, class_business, class_eco, class_eco_plus):
    """Inserer un vol dans la table Flights."""
    sql = '''INSERT INTO Flights(Flight_Distance, Class_Business, Class_Eco, Class_Eco_Plus)
             VALUES(?, ?, ?, ?)'''
    cur = conn.cursor()
    cur.execute(sql, (flight_distance, class_business, class_eco, class_eco_plus))
    conn.commit()
    return cur.lastrowid

# function de insertion table travel
def insert_travel(conn, customer_id, type_of_travel_business, type_of_travel_personal, departure_delay, arrival_delay):
    """Inserer un voyage dans la table Travel."""
    sql = '''INSERT INTO Travel(customer_id, Type_of_Travel_Business_Travel, Type_of_Travel_Personal_Travel, Departure_Delay_in_Minutes, Arrival_Delay_in_Minutes)
             VALUES(?, ?, ?, ?, ?)'''
    cur = conn.cursor()
    cur.execute(sql, (customer_id, type_of_travel_business, type_of_travel_personal, departure_delay, arrival_delay))
    conn.commit()
    return cur.lastrowid

# function de insertion table service
def insert_service(conn, travel_id, inflight_wifi_service, departure_arrival_time_convenient,
                   ease_of_online_booking, gate_location, food_and_drink, online_boarding,
                   seat_comfort, inflight_entertainment, on_board_service, leg_room_service,
                   baggage_handling, checkin_service, inflight_service, cleanliness):
    """Inserer un service dans la table Services."""
    sql = '''INSERT INTO Services(travel_id, inflight_wifi_service, departure_arrival_time_convenient,
                                   ease_of_online_booking, gate_location, food_and_drink,
                                   online_boarding, seat_comfort, inflight_entertainment,
                                   on_board_service, leg_room_service, baggage_handling,
                                   checkin_service, inflight_service, cleanliness)
             VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
    cur = conn.cursor()
    cur.execute(sql, (travel_id, inflight_wifi_service, departure_arrival_time_convenient,
                      ease_of_online_booking, gate_location, food_and_drink, online_boarding,
                      seat_comfort, inflight_entertainment, on_board_service, leg_room_service,
                      baggage_handling, checkin_service, inflight_service, cleanliness))
    conn.commit()
    return cur.lastrowid

# function de insertion table satisfaction
def insert_satisfaction(conn, travel_id, satisfaction):
    """Inserer une satisfaction dans la table Satisfaction."""
    sql = '''INSERT INTO Satisfaction(travel_id, Satisfaction)
             VALUES(?, ?)'''
    cur = conn.cursor()
    cur.execute(sql, (travel_id, satisfaction))
    conn.commit()
    return cur.lastrowid

print(24*('*'))

# Insérer les tables Customers
for index, row in data.iterrows():
    customer_id = insert_customer(conn, row['Gender_Female'], row['Gender_Male'], 
                                   row['Customer Type_Loyal Customer'], row['Customer Type_disloyal Customer'], 
                                   row['Age'])

# Insérer les tables Flights
for index, row in data.iterrows():
    flight_id = insert_flight(conn, row['Flight Distance'], row['Class_Business'], 
                               row['Class_Eco'], row['Class_Eco Plus'])

# Insérer les tables Travel
for index, row in data.iterrows():
    travel_id = insert_travel(conn, customer_id, 
                               row['Type of Travel_Business travel'], 
                               row['Type of Travel_Personal Travel'], 
                               row['Departure Delay in Minutes'], 
                               row['Arrival Delay in Minutes'])

# Insérer les tables Services
for index, row in data.iterrows():
    service_id = insert_service(conn, travel_id, 
                                row['Inflight wifi service'], 
                                row['Departure/Arrival time convenient'], 
                                row['Ease of Online booking'], 
                                row['Gate location'], 
                                row['Food and drink'], 
                                row['Online boarding'], 
                                row['Seat comfort'], 
                                row['Inflight entertainment'], 
                                row['On-board service'], 
                                row['Leg room service'], 
                                row['Baggage handling'], 
                                row['Checkin service'], 
                                row['Inflight service'], 
                                row['Cleanliness'])

# Insérer les tables Satisfaction
for index, row in data.iterrows():
    satisfaction_id = insert_satisfaction(conn, travel_id, row['Satisfaction'])
