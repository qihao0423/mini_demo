from pymongo import MongoClient

client = MongoClient()
db = client.mydb
my_set = db.my_set
my_set.remove()
for i in my_set.find():
    print(i)