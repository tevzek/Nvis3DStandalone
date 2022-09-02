import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String





rospy.init_node('ImagePrinter', anonymous=True)
rate = rospy.Rate(10)  # 3hz

pubAct = rospy.Publisher('/camera', String, queue_size=1)

while True:
    pubAct.publish("Hello world")
    rate.sleep()



