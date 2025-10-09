CREATE DATABASE StockData;
USE StockData;
CREATE TABLE StockPrices (
  symbol VARCHAR(10) NOT NULL,
  open    DECIMAL(10,5),
  high    DECIMAL(10,5),
  low     DECIMAL(10,5),
  ts      DATETIME    NOT NULL,
  PRIMARY KEY (symbol, ts)
);
ALTER TABLE StockPrices 
  CHANGE COLUMN `open` open_price DECIMAL(10,5),
  CHANGE COLUMN `high` high_price DECIMAL(10,5),
  CHANGE COLUMN `low`  low_price  DECIMAL(10,5);
SELECT * FROM StockPrices;