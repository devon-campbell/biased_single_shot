import numpy as np
from tqdm import tqdm
import json
import time
import datetime
from ldpc import BpOsdDecoder
from bposd.css import css_code
import scipy
from lifted_hgp_4d import *
from collections import Counter

class css_ss_decode_sim:
    def __init__(self, hx=None, hz=None, mx=None, mz=None, **input_dict):
        # default input values
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
            "hadamard_rotate": False,
            "hadamard_rotate_sector1_length": 0,
            "run_ss": False, # when True, meas errs occur
            "apply_ss": True, # when True (and run_ss is True), attempt to apply corrections based on metachecks
            "p_meas_err": 0,
            "error_bar_precision_cutoff": 1e-3,
            "run_sustained": False,      # run the sustained-threshold experiment?
            "sustained_threshold_depth": 4,  # how many repeated measurement rounds to do
        }

        # apply defaults for keys not passed to the class
        for key in input_dict.keys():
            self.__dict__[key] = input_dict[key]
        for key in default_input.keys():
            if key not in input_dict:
                self.__dict__[key] = default_input[key]

        # output variables
        output_values = {
            "K": None,
            "N": None,
            "start_date": None,
            "runtime": 0.0,
            "runtime_readable": None,
            "run_count": 0,
            "bp_converge_count_x": 0,
            "bp_converge_count_z": 0,
            "bp_success_count": 0,
            "bp_logical_error_rate": 0,
            "bp_logical_error_rate_eb": 0,
            "osd0_success_count": 0,
            "osd0_logical_error_rate": 0.0,
            "osd0_logical_error_rate_eb": 0.0,
            "osdw_success_count": 0,
            "osdw_logical_error_rate": 0.0,
            "osdw_logical_error_rate_eb": 0.0,
            "osdw_word_error_rate": 0.0,
            "osdw_word_error_rate_eb": 0.0,
            "min_logical_weight": 1e9,
        }

        for key in output_values.keys():  # copies initial values for output attributes
            if key not in self.__dict__:
                self.__dict__[key] = output_values[key]

        # the attributes we wish to save to file
        temp = []
        for key in self.__dict__.keys():
            if key not in [
                "channel_probs_x",
                "channel_probs_z",
                "channel_probs_y",
                "hx",
                "hz",
                "mx",
                "mz"
            ]:
                temp.append(key)
        self.output_keys = temp

        # random number generator setup
        if self.seed == 0 or self.run_count != 0:
            self.seed = np.random.randint(low=1, high=2**32 - 1)
        np.random.seed(self.seed)
        print(f"RNG Seed: {self.seed}")

        # the hx, hx, mx, mz matrices
        self.hx = scipy.sparse.csr_matrix(hx).astype(np.uint8)
        self.hz = scipy.sparse.csr_matrix(hz).astype(np.uint8)

        # print(type(mx))
        self.mx = scipy.sparse.csr_matrix(mx).astype(np.uint8)
        self.mz = scipy.sparse.csr_matrix(mz).astype(np.uint8)
        self.N = self.hz.shape[1]  # the block length
        if (
            self.min_logical_weight == 1e9
        ):  # the minimum observed weight of a logical operator
            self.min_logical_weight = self.N
        self.error_x = np.zeros(self.N).astype(np.uint8)  # x_component error vector
        self.error_z = np.zeros(self.N).astype(np.uint8)  # z_component error vector

        # construct the CSS code from hx and hz
        self._construct_code()

        # setup the error channel
        self._error_channel_setup()

        # setup the BP+OSD decoders
        self._decoder_setup()

        if self.run_sim:
            self.run_decode_sim()

    def _single_run(self):
        # randomly generate the error
        self.error_x, self.error_z = self._generate_error()

        if self.channel_update is None:
            # decode z
            synd_z = self.hx @ self.error_z % 2
            self.bpd_z.decode(synd_z)

            # decode x
            synd_x = self.hz @ self.error_x % 2
            self.bpd_x.decode(synd_x)

        elif self.channel_update == "z->x":
            # decode z
            synd_z = self.hx @ self.error_z % 2
            self.bpd_z.decode(synd_z)

            # update the channel probability
            self._channel_update(self.channel_update)

            # decode x
            synd_x = self.hz @ self.error_x % 2
            self.bpd_x.decode(synd_x)

        elif self.channel_update == "x->z":
            # decode x
            synd_x = self.hz @ self.error_x % 2
            self.bpd_x.decode(synd_x)

            # update the channel probability
            self._channel_update(self.channel_update)

            # decode z
            synd_z = self.hx @ self.error_z % 2
            self.bpd_z.decode(synd_z)

        # compute the logical and word error rates
        self._encoded_error_rates()

    def _single_run_ss(self):
        # ─────────────────────────────────────────────────────────────────────
        # 1) Generate physical data error
        self.error_x, self.error_z = self._generate_error()
    
        # 2) Compute ideal syndromes
        synd_x_ideal = (self.hz @ self.error_x) % 2
        synd_z_ideal = (self.hx @ self.error_z) % 2
    
        # 3) Add measurement noise
        synd_x_noisy = (synd_x_ideal + np.random.binomial(1, self.p_meas_err, len(synd_x_ideal))) % 2
        synd_z_noisy = (synd_z_ideal + np.random.binomial(1, self.p_meas_err, len(synd_z_ideal))) % 2
    
        # Flags for actual measurement errors (any syndrome bit flipped)
        err_meas_x = not np.array_equal(synd_x_ideal, synd_x_noisy)
        err_meas_z = not np.array_equal(synd_z_ideal, synd_z_noisy)
    
        # ─────────────────────────────────────────────────────────────────────
        # 4) Compute meta-syndromes
        meta_synd_x = (self.mx @ synd_x_noisy) % 2
        meta_synd_z = (self.mz @ synd_z_noisy) % 2
    
        det_x = np.any(meta_synd_x)
        det_z = np.any(meta_synd_z)
    
        # ─────────────────────────────────────────────────────────────────────
        # 5) Decode measurement layer
        self.bpd_meas_x.decode(meta_synd_x)
        self.bpd_meas_z.decode(meta_synd_z)
    
        meas_err_correction_x = self.bpd_meas_x.osdw_decoding
        meas_err_correction_z = self.bpd_meas_z.osdw_decoding
    
        # 6) Apply the correction
        synd_x_corrected = (synd_x_noisy + meas_err_correction_x) % 2
        synd_z_corrected = (synd_z_noisy + meas_err_correction_z) % 2

        synd_x_corrected = synd_x_corrected if self.apply_ss else synd_x_noisy
        synd_x_corrected = synd_x_corrected if self.apply_ss else synd_z_noisy
    
        fix_x = np.array_equal(synd_x_corrected, synd_x_ideal)
        fix_z = np.array_equal(synd_z_corrected, synd_z_ideal)
    
        # ─────────────────────────────────────────────────────────────────────
        # 7) Decode corrected syndromes for data error
        if self.channel_update is None:
            self.bpd_z.decode(synd_z_corrected)
            self.bpd_x.decode(synd_x_corrected)
        elif self.channel_update == "z->x":
            self.bpd_z.decode(synd_z_corrected)
            self._channel_update("z->x")
            self.bpd_x.decode(synd_x_corrected)
        elif self.channel_update == "x->z":
            self.bpd_x.decode(synd_x_corrected)
            self._channel_update("x->z")
            self.bpd_z.decode(synd_z_corrected)
    
        # 8) Evaluate final success/failure
        logical_success = self._encoded_error_rates()
    
        # ─────────────────────────────────────────────────────────────────────
        # Collect detailed summary statistics
        if not hasattr(self, "stats"):
            self.stats = Counter(trials=0,
                                 TPx=0, FNx=0, FPx=0, TNx=0, TCx=0, MCx=0,
                                 TPz=0, FNz=0, FPz=0, TNz=0, TCz=0, MCz=0,
                                 logical_fail=0)
    
        S = self.stats
        S["trials"] += 1
    
        # -- X syndrome stats
        if err_meas_x and det_x:     S["TPx"] += 1
        if err_meas_x and not det_x: S["FNx"] += 1
        if not err_meas_x and det_x: S["FPx"] += 1
        if not err_meas_x and not det_x: S["TNx"] += 1
        if err_meas_x and fix_x:     S["TCx"] += 1
        if det_x and not fix_x:      S["MCx"] += 1
    
        # -- Z syndrome stats
        if err_meas_z and det_z:     S["TPz"] += 1
        if err_meas_z and not det_z: S["FNz"] += 1
        if not err_meas_z and det_z: S["FPz"] += 1
        if not err_meas_z and not det_z: S["TNz"] += 1
        if err_meas_z and fix_z:     S["TCz"] += 1
        if det_z and not fix_z:      S["MCz"] += 1
    
        # -- Final logical success/failure
        if not logical_success:
            S["logical_fail"] += 1

    def print_ss_summary(self):
        S = self.stats
        for basis in ("x", "z"):
            TP = S[f"TP{basis}"]; FN = S[f"FN{basis}"]
            FP = S[f"FP{basis}"]; TN = S[f"TN{basis}"]
            TC = S[f"TC{basis}"]; MC = S[f"MC{basis}"]
    
            det_rate   = TP / (TP + FN) if TP+FN else 0.0
            false_pos  = FP / (FP + TN) if FP+TN else 0.0
            corr_rate  = TC / (TP + FN) if TP+FN else 0.0
            miscorr    = MC / (TP + FP) if TP+FP else 0.0
    
            print(f"\n--- {basis.upper()} basis ---")
            print(f"  measurement errors occurred : {TP+FN}")
            print(f"  detection rate              : {det_rate:6.2%}")
            print(f"  false-alarm rate            : {false_pos:6.2%}")
            print(f"  correction success (TP)     : {corr_rate:6.2%}")
            print(f"  mis-correction given detect : {miscorr:6.2%}")
    
        wer = S["logical_fail"] / S["trials"]
        print(f"\nPost‑decode logical WER : {wer:6.2%}  (over {S['trials']} trials)")
    
    def _channel_update(self, update_direction):
        """
        Function updates the channel probability vector for the second decoding component
        based on the first. The channel probability updates can be derived from Bayes' rule.
        """

        # x component first, then z component
        if update_direction == "x->z":
            decoder_probs = np.zeros(self.N)
            for i in range(self.N):
                if self.bpd_x.osdw_decoding[i] == 1:
                    if (self.channel_probs_x[i] + self.channel_probs_y[i]) == 0:
                        decoder_probs[i] = 0
                    else:
                        decoder_probs[i] = self.channel_probs_y[i] / (
                            self.channel_probs_x[i] + self.channel_probs_y[i]
                        )
                elif self.bpd_x.osdw_decoding[i] == 0:
                    decoder_probs[i] = self.channel_probs_z[i] / (
                        1 - self.channel_probs_x[i] - self.channel_probs_y[i]
                    )

            self.bpd_z.update_channel_probs(decoder_probs)

        # z component first, then x component
        elif update_direction == "z->x":
            self.bpd_z.osdw_decoding
            decoder_probs = np.zeros(self.N)
            for i in range(self.N):
                if self.bpd_z.osdw_decoding[i] == 1:
                    if (self.channel_probs_z[i] + self.channel_probs_y[i]) == 0:
                        decoder_probs[i] = 0
                    else:
                        decoder_probs[i] = self.channel_probs_y[i] / (
                            self.channel_probs_z[i] + self.channel_probs_y[i]
                        )
                elif self.bpd_z.osdw_decoding[i] == 0:
                    decoder_probs[i] = self.channel_probs_x[i] / (
                        1 - self.channel_probs_z[i] - self.channel_probs_y[i]
                    )

            self.bpd_x.update_channel_probs(decoder_probs)

    def _encoded_error_rates(self):
        """
        Updates the logical and word error rates for OSDW, OSD0 and BP (before post-processing)
        """
        # OSDW Logical error rate
        # calculate the residual error
        residual_x = (self.error_x + self.bpd_x.osdw_decoding) % 2
        residual_z = (self.error_z + self.bpd_z.osdw_decoding) % 2

        # check for logical X-error
        if (self.lz @ residual_x % 2).any():
            logical_weight = np.sum(residual_x)
            if logical_weight < self.min_logical_weight:
                self.min_logical_weight = int(logical_weight)

        # check for logical Z-error
        elif (self.lx @ residual_z % 2).any():
            logical_weight = np.sum(residual_z)
            if logical_weight < self.min_logical_weight:
                self.min_logical_weight = int(logical_weight)
        else:
            self.osdw_success_count += 1

        # compute logical error rate
        self.osdw_logical_error_rate = 1 - self.osdw_success_count / self.run_count
        self.osdw_logical_error_rate_eb = np.sqrt(
            (1 - self.osdw_logical_error_rate)
            * self.osdw_logical_error_rate
            / self.run_count
        )

        # compute word error rate
        self.osdw_word_error_rate = 1.0 - (1 - self.osdw_logical_error_rate) ** (
            1 / self.K
        )
        self.osdw_word_error_rate_eb = (
            self.osdw_logical_error_rate_eb
            * ((1 - self.osdw_logical_error_rate_eb) ** (1 / self.K - 1))
            / self.K
        )

        # OSD0 logical error rate
        # calculate the residual error
        residual_x = (self.error_x + self.bpd_x.osd0_decoding) % 2
        residual_z = (self.error_z + self.bpd_z.osd0_decoding) % 2

        # check for logical X-error
        if (self.lz @ residual_x % 2).any():
            logical_weight = np.sum(residual_x)
            if logical_weight < self.min_logical_weight:
                self.min_logical_weight = int(logical_weight)

        # check for logical Z-error
        elif (self.lx @ residual_z % 2).any():
            logical_weight = np.sum(residual_z)
            if logical_weight < self.min_logical_weight:
                self.min_logical_weight = int(logical_weight)
        else:
            self.osd0_success_count += 1

        # compute logical error rate
        self.osd0_logical_error_rate = 1 - self.osd0_success_count / self.run_count
        self.osd0_logical_error_rate_eb = np.sqrt(
            (1 - self.osd0_logical_error_rate)
            * self.osd0_logical_error_rate
            / self.run_count
        )

        # compute word error rate
        self.osd0_word_error_rate = 1.0 - (1 - self.osd0_logical_error_rate) ** (
            1 / self.K
        )
        self.osd0_word_error_rate_eb = (
            self.osd0_logical_error_rate_eb
            * ((1 - self.osd0_logical_error_rate_eb) ** (1 / self.K - 1))
            / self.K
        )

        # BP Logical error rate
        # check for convergence
        if self.bpd_z.converge:
            self.bp_converge_count_z += 1
        if self.bpd_x.converge:
            self.bp_converge_count_x += 1

        if self.bpd_z.converge and self.bpd_x.converge:
            # calculate the residual error
            residual_x = (self.error_x + self.bpd_x.bp_decoding) % 2
            residual_z = (self.error_z + self.bpd_z.bp_decoding) % 2

            # check for logical X-error
            if (self.lz @ residual_x % 2).any():
                pass

            # check for logical Z-error
            elif (self.lx @ residual_z % 2).any():
                pass
            else:
                self.bp_success_count += 1

        # compute logical error rate
        self.bp_logical_error_rate = 1 - self.bp_success_count / self.run_count
        self.bp_logical_error_rate_eb = np.sqrt(
            (1 - self.bp_logical_error_rate)
            * self.bp_logical_error_rate
            / self.run_count
        )

        # compute word error rate
        self.bp_word_error_rate = 1.0 - (1 - self.bp_logical_error_rate) ** (1 / self.K)
        self.bp_word_error_rate_eb = (
            self.bp_logical_error_rate_eb
            * ((1 - self.bp_logical_error_rate_eb) ** (1 / self.K - 1))
            / self.K
        )

    def _construct_code(self):
        """
        Constructs the CSS code from the hx and hz stabilizer matrices.
        """

        print("Constructing CSS code from hx and hz matrices...")
        if isinstance(self.hx, (np.ndarray, scipy.sparse.spmatrix)) and isinstance(
            self.hz, (np.ndarray, scipy.sparse.spmatrix)
        ):
            print(f'h shapes: {self.hx.shape, self.hz.shape}')
            qcode = css_code(self.hx, self.hz)
            self.lx = qcode.lx
            self.lz = qcode.lz
            print(f'l shapes: {self.lx.shape, self.lz.shape}')
            self.K = qcode.K
            self.N = qcode.N
            print(f'K: {qcode.K}')
            print(f'N: {qcode.N}')
            print("Checking the CSS code is valid...")
            if self.check_code and not qcode.test():
                raise Exception(
                    "Error: invalid CSS code. Check the form of your hx and hz matrices!"
                )
        else:
            raise Exception("Invalid object type for the hx/hz matrices")
        return None

    def _error_channel_setup(self):
        """
        Sets up the error channels from the error rate and error bias input parameters
        """

        xyz_error_bias = np.array(self.xyz_error_bias)
        if xyz_error_bias[0] == np.inf:
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
        else:
            self.px, self.py, self.pz = (
                self.error_rate * xyz_error_bias / np.sum(xyz_error_bias)
            )

        if self.hadamard_rotate == 0:
            self.channel_probs_x = np.ones(self.N) * (self.px)
            self.channel_probs_z = np.ones(self.N) * (self.pz)
            self.channel_probs_y = np.ones(self.N) * (self.py)

        elif self.hadamard_rotate == 1:
            n1 = self.hadamard_rotate_sector1_length
            self.channel_probs_x = np.hstack(
                [np.ones(n1) * (self.px), np.ones(self.N - n1) * (self.pz)]
            )
            self.channel_probs_z = np.hstack(
                [np.ones(n1) * (self.pz), np.ones(self.N - n1) * (self.px)]
            )
            self.channel_probs_y = np.ones(self.N) * (self.py)
        else:
            raise ValueError(
                f"The hadamard rotate attribute should be set to 0 or 1. Not '{self.hadamard_rotate}"
            )

        self.channel_probs_x.setflags(write=False)
        self.channel_probs_y.setflags(write=False)
        self.channel_probs_z.setflags(write=False)

    def _decoder_setup(self):
        """
        Setup for the BP+OSD decoders for data errors *and* measurement errors.
        """
    
        self.ms_scaling_factor = float(self.ms_scaling_factor)
    
        # --- 1) Data-error decoders ---
        # For Z errors, we use hx:
        self.bpd_z = BpOsdDecoder(
            self.hx,
            channel_probs=self.channel_probs_z + self.channel_probs_y,
            max_iter=self.max_iter,
            bp_method=self.bp_method,
            ms_scaling_factor=self.ms_scaling_factor,
            osd_method=self.osd_method,
            osd_order=self.osd_order,
        )
        # For X errors, we use hz:
        self.bpd_x = BpOsdDecoder(
            self.hz,
            channel_probs=self.channel_probs_x + self.channel_probs_y,
            max_iter=self.max_iter,
            bp_method=self.bp_method,
            ms_scaling_factor=self.ms_scaling_factor,
            osd_method=self.osd_method,
            osd_order=self.osd_order,
        )
    
        # --- 2) Measurement-error decoders (new!) ---
        # Only define them if mx/mz are provided & not empty
        if self.mx is not None and self.mx.shape[0] != 0:
            # Here, channel_probs for measurement errors can be uniform:
            meas_err_probs_x = np.ones(self.mx.shape[1]) * self.p_meas_err
            self.bpd_meas_x = BpOsdDecoder(
                self.mx,
                channel_probs=meas_err_probs_x,
                max_iter=3,
                bp_method="product_sum",   
                ms_scaling_factor=self.ms_scaling_factor,
                osd_order=0
            )
    
        if self.mz is not None and self.mz.shape[0] != 0:
            meas_err_probs_z = np.ones(self.mz.shape[1]) * self.p_meas_err
            self.bpd_meas_z = BpOsdDecoder(
                self.mz,
                channel_probs=meas_err_probs_z,
                max_iter=3,
                bp_method="product_sum",   
                ms_scaling_factor=self.ms_scaling_factor,
                osd_order=0
            )



    def _generate_error(self):
        """
        Generates a random error on both the X and Z components of the code
        distributed according to the channel probability vectors.
        """

        for i in range(self.N):
            rand = np.random.random()
            if rand < self.channel_probs_z[i]:
                self.error_z[i] = 1
                self.error_x[i] = 0
            elif (
                self.channel_probs_z[i]
                <= rand
                < (self.channel_probs_z[i] + self.channel_probs_x[i])
            ):
                self.error_z[i] = 0
                self.error_x[i] = 1
            elif (
                (self.channel_probs_z[i] + self.channel_probs_x[i])
                <= rand
                < (
                    self.channel_probs_x[i]
                    + self.channel_probs_y[i]
                    + self.channel_probs_z[i]
                )
            ):
                self.error_z[i] = 1
                self.error_x[i] = 1
            else:
                self.error_z[i] = 0
                self.error_x[i] = 0

        return self.error_x, self.error_z

    def run_decode_sim(self):
        """
        This function contains the main simulation loop and controls the output.
        """

        # save start date
        self.start_date = datetime.datetime.fromtimestamp(time.time()).strftime(
            "%A, %B %d, %Y %H:%M:%S"
        )

        pbar = tqdm(
            range(self.run_count + 1, self.target_runs + 1),
            disable=self.tqdm_disable,
            ncols=0,
        )

        start_time = time.time()
        save_time = start_time

        for self.run_count in pbar:
            if self.run_sustained:
                self._single_run_sustained()   
            elif self.run_ss:
                self._single_run_ss()
            else:
                self._single_run()

            pbar.set_description(
                f"d_max: {self.min_logical_weight}; OSDW_WER: {self.osdw_word_error_rate*100:.3g}±{self.osdw_word_error_rate_eb*100:.2g}%; OSDW: {self.osdw_logical_error_rate*100:.3g}±{self.osdw_logical_error_rate_eb*100:.2g}%; OSD0: {self.osd0_logical_error_rate*100:.3g}±{self.osd0_logical_error_rate_eb*100:.2g}%;"
            )

            current_time = time.time()
            save_loop = current_time - save_time

            if (
                int(save_loop) > self.save_interval
                or self.run_count == self.target_runs
            ):
                save_time = time.time()
                self.runtime = save_loop + self.runtime

                self.runtime_readable = time.strftime(
                    "%H:%M:%S", time.gmtime(self.runtime)
                )

                if self.output_file is not None:
                    f = open(self.output_file, "w+")
                    print(self.output_dict(), file=f)
                    f.close()

                if (
                    self.osdw_logical_error_rate_eb > 0
                    and self.osdw_logical_error_rate_eb / self.osdw_logical_error_rate
                    < self.error_bar_precision_cutoff
                ):
                    print(
                        "\nTarget error bar precision reached. Stopping simulation..."
                    )
                    break

        if self.run_ss or self.run_sustained:
            self.print_ss_summary()
        return json.dumps(self.output_dict(), sort_keys=True, indent=4)

    def output_dict(self):
        """
        Function for formatting the output
        """

        output_dict = {}
        for key, value in self.__dict__.items():
            if key in self.output_keys:
                output_dict[key] = value
        # return output_dict
        return json.dumps(output_dict, sort_keys=True, indent=4)