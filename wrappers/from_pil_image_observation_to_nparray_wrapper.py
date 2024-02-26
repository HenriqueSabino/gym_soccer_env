from PIL import Image
import numpy as np
from gymnasium import ObservationWrapper

class SoccerEnvImageObservation2NpArrayWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        assert env.unwrapped.observation_type == 'image', "Tipo de observação deve ser 'image'"

    def observation(self, observation: Image) -> np.ndarray:

        return np.array(
            self.observation.getdata(),
            dtype=np.uint8
        ).reshape(self.observation.size[0], self.observation.size[1], 3)
