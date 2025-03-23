# this code is originally from css_decode_sim.py

default_input = {
    "error_rate": None,
    "xyz_error_bias": [1, 1, 1],
    "target_runs": 100,
    "seed": 0,
    "bp_method": "minimum_sum",
    "ms_scaling_factor": 0.625,
    "max_iter": 0,
    "osd_method": "osd_cs",
    "osd_order": 2,
    "save_interval": 2,
    "output_file": None,
    "check_code": 1,
    "tqdm_disable": 0,
    "run_sim": 1,
    "channel_update": "x->z",
    "sector_list": [0, 1, 2],
    "rotated_sectors": [],
    "sector_lengths": {0:0, 1:0, 2:0},
    "error_bar_precision_cutoff": 1e-3,
}

    def _error_channel_setup(self):
        """
        Sets up the error channels from the error rate and error bias input parameters
        """
    
        # SET UP ERR RATE FOR EACH PAULI (DEPENDING ON BIAS):
        xyz_error_bias = np.array(self.xyz_error_bias)  # take in the tuple with (xbias, ybias, zbias)
        if xyz_error_bias[0] == np.inf:                 # if infinite bias, that Pauli is the whole err rate (everything else = 0)
            self.px = self.error_rate
            self.py = 0
            self.pz = 0
        elif xyz_error_bias[1] == np.inf:
            self.px = 0
            self.py = self.error_rate
            self.pz = 0
        elif xyz_error_bias[2] == np.inf:
            self.px = 0
            self.py = 0
            self.pz = self.error_rate
        else:                                           # otherwise, each Pauli gets the proportional error prob
            self.px, self.py, self.pz = (
                self.error_rate * xyz_error_bias / np.sum(xyz_error_bias)
            )
    
        # SET UP CHANNEL PROBS (DEPENDING ON H ROT)
        if not self.rotated_sectors:       
            # No sectors are rotated → apply uniform px, pz
            self.channel_probs_x = np.ones(self.N) * self.px
            self.channel_probs_z = np.ones(self.N) * self.pz
            self.channel_probs_y = np.ones(self.N) * self.py
        
        else:
            # Some sectors are rotated → build channel_probs accordingly
            x_probs, z_probs = [], []
            for i in range(len(self.sector_lengths)):  # supports any number of sectors
                n = self.sector_lengths[i]
                if i in self.rotated_sectors:
                    # Hadamard-rotated sector: swap px ↔ pz
                    x_probs.append(np.ones(n) * self.pz)
                    z_probs.append(np.ones(n) * self.px)
                else:
                    # Unrotated sector: standard assignment
                    x_probs.append(np.ones(n) * self.px)
                    z_probs.append(np.ones(n) * self.pz)
        
            self.channel_probs_x = np.hstack(x_probs)
            self.channel_probs_z = np.hstack(z_probs[::-1]) # mirror, due to mirrored sector structure for Hz
            self.channel_probs_y = np.ones(self.N) * self.py
    
        self.channel_probs_x.setflags(write=False)
        self.channel_probs_y.setflags(write=False)
        self.channel_probs_z.setflags(write=False)