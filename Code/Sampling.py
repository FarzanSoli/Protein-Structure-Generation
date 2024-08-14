import copy
import torch
from Config import config
# ====================================================
class Diffusion_Process():
    # ================================================
    def __init__(self, trained_model, coordinates_shape, features_shape, eta=0, tau=1):
        self.eta = eta
        self.tau = tau
        self.sigmas = config().sigmas
        self.features_shape = features_shape
        self.alpha_bars = config().alpha_bars
        self.coordinates_shape = coordinates_shape
        # -----------------------------------
        # Reverse Diffusion (EGNN)
        self.model = trained_model
    # ==================================================== # 
    #                     Beta Scheduling                  #
    # ==================================================== #    
    def diffusion_time_steps(self, reverse=True):
        # tau -> tau = 50 -> samples every 50 steps and accelerates diffusion.
        # [len(self.alpha_bars)-1] --> last index
        diffusion_process=list(
            range(0,len(self.alpha_bars),self.tau))+[len(self.alpha_bars)-1]
        # ------------------------------------
        # Forward diffusion steps (0 -> T)
        if not reverse:
            diffusion_process = zip(diffusion_process[1:], diffusion_process[:-1])
        # Reverse diffusion steps (T -> 0)
        if reverse:
            diffusion_process = zip(reversed(diffusion_process[:-1]), 
                                    reversed(diffusion_process[1:]))
        return diffusion_process
    # ==================================================== # 
    #                Reverse Diffusion Coordinates         #
    # ==================================================== #
    def reverse_diffusion_steps(self, x_t, f_t, reverse=True):
        seq_len = config().num_residues
        batch_size = config().batch_size
        diffusion_process = self.diffusion_time_steps(reverse)
        sample_features = {'residue_index': torch.tile(
                            torch.arange(seq_len), (batch_size, 1))}
        sample_coordinates = {'residue_index': torch.tile(
                            torch.arange(seq_len), (batch_size, 1))}
        for prev_t, current_t in diffusion_process:
            self.model.eval()
            # ------------------------------------ #
            #  Physicochemical Features Corrupted  #
            # ------------------------------------ #
            sample_features['bb_corrupted'] = copy.copy(f_t)
            sample_features = {k: v.to(config().device) for k, v in sample_features.items()}
            # ------------------------------------ #
            #        Coordinates Corrupted         #
            # ------------------------------------ #
            sample_coordinates['bb_corrupted'] = copy.copy(x_t)
            sample_coordinates = {k: v.to(config().device) for k, v in sample_coordinates.items()}
            # ------------------------------------ #
            #       Predict Epsilon (Noise)        #
            # ------------------------------------ #
            # [batch_size, residue_length_ 3]
            eps_theta_x, eps_theta_f = self.model(sample_coordinates['bb_corrupted'], 
                                    sample_features['bb_corrupted'])
            # zero noise at t=0 and for the rest noise is shaped like x_t
            noise_x = torch.zeros_like(x_t) if current_t == 0 else torch.randn_like(x_t)
            sigma = self.sigmas[current_t] * self.eta
            # --------------------------
            # Using Denoizer model to predict original data (x_0) - eq.15 DDPM paper.
            pred_x_0 = (x_t-torch.sqrt(1-self.alpha_bars[current_t])*eps_theta_x)/\
                            torch.sqrt(self.alpha_bars[current_t])
            # ------------------------------------ #
            #           DDIM Formulation           #
            # ------------------------------------ #
            direction_to_x_t = torch.sqrt(1-self.alpha_bars[prev_t]-sigma**2)*eps_theta_x
            # ------------------------------------
            x_t = torch.sqrt(self.alpha_bars[prev_t])*pred_x_0+direction_to_x_t+sigma*noise_x
            # ==================================== #
            # zero noise at t=0 and for the rest noise is shaped like x_t
            noise_f = torch.zeros_like(f_t) if current_t == 0 else torch.randn_like(f_t)
            # ------------------------------------
            # Using Denoizer model to predict original data (f_0)
            pred_f_0 = (f_t-torch.sqrt(1-self.alpha_bars[current_t])*eps_theta_f)/\
                            torch.sqrt(self.alpha_bars[current_t])
            # ------------------------------------ #
            #           DDIM Formulation           #
            # ------------------------------------ #
            direction_to_f_t = torch.sqrt(1-self.alpha_bars[prev_t]-sigma**2)*eps_theta_f
            # ------------------------------------
            f_t = torch.sqrt(self.alpha_bars[prev_t])*pred_f_0+direction_to_f_t+sigma*noise_f
        yield x_t, f_t
    # ==================================================== # 
    #              Generation Process (Sampling)           #
    # ==================================================== #
    @torch.no_grad()
    # Final returns the predicted x_0
    def Sampling(self, sample_coordinate=None, sample_features=None, only_final=False):
        """ only_final      : If True, return is an only output of final schedule step 
            *arg --> accept a changeable number of arguments """
        if sample_coordinate==None and sample_features==None:
            sample_features = torch.randn(
                                [*self.features_shape]).to(device = config().device)
            sample_coordinate = torch.randn(
                                [*self.coordinates_shape]).to(device = config().device)
        # -----------------------------------------     
        final = None
        sampling_list = []
        for sample in self.reverse_diffusion_steps(sample_coordinate, sample_features):
            # sample = (coordinates, features)
            final = sample
            if not only_final:
                sampling_list.append(final)
        # Concatenates a sequence of tensors along a new dimension.
        return self.process_bb_pos(final) if only_final else torch.stack(sampling_list)
    # ================================================
    def process_bb_pos(self, sample):
        def _process(x):
            x_center = torch.mean(x, axis=0)
            return (x - x_center)
        bb_pos_processed = torch.stack(
                        [_process(bb_pos_) for bb_pos_ in sample[0]], axis=0)
        bb_feat_processed = torch.stack(
                        [_process(bb_feat_) for bb_feat_ in sample[1]], axis=0)
        return bb_pos_processed, bb_feat_processed
    # ================================================

 