CARRY = ["carry","Carry","bring"]
PICK = ["pick","take","Take","Pick","Grab","grab","get","Get"] # ! Get can also be used as a pointer for navigation- eg - get to the desk near bed
WALK = ['go','walk','move','take']
WALK_ST = ['forward','straight']



#from navigation signatures
CONFUSIONS = {"Cabinet":["Dresser","SideTable"],"Dresser":["Cabinet","SideTable","Desk","Bureau"], "table":["Desk","SideTable","DiningTable","CoffeeTable"], 
                "SideTable":["Dresser"], "Counter":["Shelf"], "Sofa":["ArmChair"], "TVStand":["Dresser"], "Desk":["DiningTable"],
                "DeskLamp":["FloorLamp"]} #306-3 counter along the window ledge

ALTERNATIVES = {"SideTable":["Desk","Dresser"]}

GOTOOBJS = ["SideTable","Dresser","Desk","table","Shelf","DeskLamp","Bed","table","TVStand","Television","Sofa", "CounterTop", "CoffeeTable","Box",
            "Ottoman","Chair","ArmChair","Sink"]

TEXTURES = ["StandardDoor"]

#from manipulation signatures
#these things can be picked up and carried by the agent
INVEN_OBJS = ["CellPhone","Pen","Pencil","TissueBox","Statue","Watch", "Bowl", "Mug", "CD", "Laptop", "BaseballBat", "AlarmClock",
                'Box', 'Cloth','Pillow', 'TennisRacket','Vase','Cup',"CreditCard","BasketBall","Book","Plate", "KeyChain", "RemoteControl",
                "Newspaper","Knife","Pot","Apple"]

OPENS = ["Drawer","Safe","Cabinet","Microwave"]

SLICES = ["Apple"]

RECEPS = ["Desk","Dresser","Shelf","Safe","SideTable","Bed","Drawer","GarbageCan",
            "table","DiningTable", "TVStand", "Sofa", "CounterTop", "CoffeeTable",
            "Chair","Box","Ottoman","Sink","SinkBasin","Pot","CoffeeMachine"] #table is not an exact object like others it needs to be resolved
#Television is not something you put objects on, but very like to be confused when someone says put on tvstand

TOGGLES = ["DeskLamp","Microwave","Faucet"]
#TOGGLES = ["FloorLamp"]
toggleable = ["lamp","Lamp",'light','Light','tablelamp']

CONFUSIONS_M = {"Cup":"Mug", "Mug":"Cup", "Plate":"Bowl",  
                "SideTable":["Dresser"], "table":["Desk","SideTable","DiningTable","CoffeeTable"],
                "TVStand":["Dresser"]} #top row for pickup objs, bottom row for receps which has a different resolution func
#things cannot be placed on Television, its generally placed on Dresser on top od which television sits

CONFLICT = {"Bowl":["Pencil","Pen"]} #Bowl cannot be placed inside pencil (however might infer this sometimes mistakenly)

#based on the programming of the ai2thor environment sometimes you cant for exmaple place object in a sink, you have to place it in a sinkbasin
#which is funny because sinkbasin is always in the sink, so there is an embedded context here
CONTEXTUALS = {"place":{"Sink":"SinkBasin"},"toggle":{"Sink":"Faucet"}}

#some manipulation slots extracted may need to be done seperately for example pick and apple and then slice it, so accordingly 
#the instruction would need to be split into partial instructions
#SPLIT_MANIPULATION_TYPES = ["pick,slice"]

