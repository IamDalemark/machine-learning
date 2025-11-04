try:
    print(0/0)
except RuntimeError as error:
    print(error)
else: 
    print("else")
finally:
    print("finally")