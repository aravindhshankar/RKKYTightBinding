x = 1e-5
match x :
    case x if (x > 1) :
        print(f"matched {x} > 1")
    case x if x > 1e-3 : 
        print(f"matched {x} > 1e-3")
    case _ :
        print(f"matched default {x}")
