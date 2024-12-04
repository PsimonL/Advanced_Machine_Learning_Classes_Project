from stronka import db


# CREATE TABLE wallet (
# 	id INTEGER NOT NULL, 
# 	user_id INTEGER NOT NULL, 
# 	currency_code VARCHAR(3) NOT NULL, 
# 	amount DECIMAL(6, 2), 
# 	transaction_at DATETIME NOT NULL, 
# 	PRIMARY KEY (id), 
# 	FOREIGN KEY(user_id) REFERENCES user (id)
# )

# CREATE TABLE user (
# 	id INTEGER NOT NULL, 
# 	username VARCHAR(30) NOT NULL, 
# 	email_address VARCHAR(50) NOT NULL, 
# 	password_hash VARCHAR(60) NOT NULL, 
# 	PRIMARY KEY (id), 
# 	UNIQUE (username), 
# 	UNIQUE (email_address)
# )
db.create_all()
