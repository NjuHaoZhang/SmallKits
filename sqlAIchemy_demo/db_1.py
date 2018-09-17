from APP import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def __repr__(self):
        return '<User %r>' % self.username

if __name__ == "__main__":

    # 创建记录并提交
    admin = User(username='admin', email='admin@example.com')
    guest = User(username='guest', email='guest@example.com')
    db.session.add(admin)
    db.session.add(guest)
    db.session.commit()

    # 查
    r = User.query.all()
    r2 = User.query.filter_by(username='admin').first()
    print(r)
    print(r2)

    # 改


    # 物理删 {最好是用改实现逻辑删除}
