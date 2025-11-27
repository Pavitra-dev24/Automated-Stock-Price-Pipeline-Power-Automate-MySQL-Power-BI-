CREATE DATABASE StockData;
USE StockData;

CREATE TABLE IF NOT EXISTS StockPrices (
    id INT AUTO_INCREMENT PRIMARY KEY,
    stock_name VARCHAR(64) NOT NULL,
    price DECIMAL(12,4) NOT NULL,
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_stock_time (stock_name, timestamp)
);

CREATE TABLE IF NOT EXISTS Predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    stock_name VARCHAR(64) NOT NULL,
    predicted_price DECIMAL(12,4) NOT NULL,
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_pred_stock_time (stock_name, timestamp)
);

CREATE TABLE IF NOT EXISTS CurrentPrices (
    id INT AUTO_INCREMENT PRIMARY KEY,
    stock_name VARCHAR(64) NOT NULL,
    price DECIMAL(12,4) NOT NULL,
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_stock_time (stock_name, timestamp)
);

select * from StockPrices;
select * from Predictions;
select * from CurrentPrices;

truncate table StockPrices;
truncate table Predictions;
truncate table CurrentPrices;

