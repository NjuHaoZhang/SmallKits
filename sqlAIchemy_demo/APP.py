from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:icanido@127.0.0.1:3306/d_test?charset=utf8'
db = SQLAlchemy(app)

# 创建表 {只有等到使用了表才会真的物理上创建}
db.create_all()

# 使用表