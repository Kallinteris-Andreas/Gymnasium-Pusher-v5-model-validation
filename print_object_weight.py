import gymnasium

#env = gymnasium.make("Pusher-v5", xml_file="./pusher_v5rc4.xml")
env = gymnasium.make("Pusher-v5")

print(f"object mass={env.unwrapped.model.body(11).mass}")
