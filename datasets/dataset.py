import roomsetup

class Dataset(object):
    def __init__(self,
                room_setup: roomsetup.RoomSetup,
                preprocess_dir: str
                ):
        self.room_setup = room_setup
        self.preprocess_dir = preprocess_dir