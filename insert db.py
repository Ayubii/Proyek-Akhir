import mysql.connector

def insertdb():

    mydb = mysql.connector.connect(
        host='localhost',
        database='absensi',
        user='coba',
        password=''
    )

    mycursor = mydb.cursor()

    sql = ("INSERT INTO `presensi`(`Nama`) VALUES (%s)")
    val = ("Muhammad Maulana Alkahfi", )
    mycursor.execute(sql, val)

    mydb.commit()

    print(mycursor.rowcount, "record inserted.")


insertdb()