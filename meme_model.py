import os

import torch
import yaml


def bernoulli_kl(logits, labels):
    """
    Compute the total KL divergence between the labels and the distributions the logits.
    :param logits: Tensor of shape (batch_size, num_classes)
    :param labels: Tensor of shape (batch_size,)
    :return: Tensor of shape (batch_size,)
    """
    original_dtype = logits.dtype
    logits = logits.float()  # Ensure logits are float for numerical stability
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    if labels is not None:
        # Average ratings across judges.
        # Just takes the mean of the ratings while accounting the fact that different questions may have different number of ratings
        labels1 = labels.float()
        labels1_mask = labels1 >= 0
        labels1 = labels1.masked_fill(~labels1_mask, 0)
        labels1 = labels1.sum(dim=-1, keepdim=True) / labels1_mask.sum(dim=-1, keepdim=True).clamp(
            min=1e-8
        )  # Avoid division by zero
        labels0 = 1 - labels1

        negative_cross_entropy = labels1 * logprobs[:, 0] + labels0 * logprobs[:, 1]
        cross_entropy = -negative_cross_entropy.to(original_dtype)
    else:
        cross_entropy = None
    return (
        cross_entropy,
        logprobs[..., 0].exp().to(original_dtype),
        torch.zeros_like(logits[..., 0]).to(original_dtype),
    )


def mse_loss(logits, labels):
    """
    Compute the mean squared error loss between the logits and the labels.
    :param logits: Tensor of shape (batch_size, num_classes)
    :param labels: Tensor of shape (batch_size,)
    :return: Tensor of shape (batch_size,)
    """
    original_dtype = logits.dtype
    logits = logits.float()  # Ensure logits are float for numerical stability
    probs = torch.nn.functional.softmax(logits, dim=-1)
    if labels is not None:
        # Average ratings across judges.
        # Just takes the mean of the ratings while accounting the fact that different questions may have different number of ratings
        labels1 = labels.float()
        labels1_mask = labels1 >= 0
        labels1 = labels1.masked_fill(~labels1_mask, 0)
        labels1 = labels1.sum(dim=-1, keepdim=True) / labels1_mask.sum(dim=-1, keepdim=True).clamp(
            min=1e-8
        )  # Avoid division by zero
        labels0 = 1 - labels1

        mse = (labels1 - probs[:, 0]) ** 2
    else:
        mse = None
    return (
        mse,
        probs[..., 0].to(original_dtype),
        torch.zeros_like(logits[..., 0]).to(original_dtype),
    )


def beta_llh(params, labels, eps=1e-2):
    """
    Compute the log likelihood of the labels given the logits for a binary classification.
    :param params: Tensor of shape (batch_size, 2)
    :param labels: Tensor of shape (batch_size,)
    :return: Tensor of shape (batch_size,)
    """
    original_dtype = params.dtype
    params: torch.Tensor = params.float()
    params = params.exp()
    concentration1 = params[:, 0]
    concentration0 = params[:, 1]
    beta_distribution = torch.distributions.Beta(
        concentration1.unsqueeze(-1), concentration0.unsqueeze(-1)
    )
    if labels is not None:
        labels = labels.clamp(min=eps, max=1 - eps)  # Avoid infs in the beta pdf
        log_probs = beta_distribution.log_prob(labels.float())
        negative_log_probs = -log_probs.to(original_dtype)
    else:
        negative_log_probs = None
    return (
        negative_log_probs,
        (concentration1 / (concentration1 + concentration0)).to(original_dtype),
        # (concentration1 - 1)/(concentration0 + concentration1 - 2).to(original_dtype),
        (concentration0 * concentration1)
        / ((concentration1 + concentration0 + 1) * ((concentration1 + concentration0) ** 2)).to(
            original_dtype
        ),
    )


def beta_moment_matching(params, labels, eps=1e-2):
    """
    Compute the log likelihood of the labels given the logits for a binary classification.
    :param params: Tensor of shape (batch_size, 2)
    :param labels: Tensor of shape (batch_size,)
    :return: Tensor of shape (batch_size,)
    """
    original_dtype = params.dtype
    params: torch.Tensor = params.float()
    params = params.exp()
    concentration1 = params[:, 0]
    concentration0 = params[:, 1]

    beta_distribution = torch.distributions.Beta(
        concentration1.unsqueeze(-1), concentration0.unsqueeze(-1)
    )
    if labels is not None:
        labels = labels.clamp(min=eps, max=1 - eps)  # Avoid infs in the beta pdf
        labels_mean = labels.mean(
            dim=-1,
        )
        labels_variance = labels.var(dim=-1, unbiased=False)
        predicted_mean = concentration1 / (concentration1 + concentration0)
        predicted_variance = (concentration0 * concentration1) / (
            (concentration1 + concentration0 + 1) * ((concentration1 + concentration0) ** 2)
        )
        moments_difference = (predicted_mean - labels_mean) ** 2 + (
            predicted_variance - labels_variance
        ) ** 2
    else:
        moments_difference = None
    return (
        moments_difference,
        (concentration1 / (concentration1 + concentration0)).to(original_dtype),
        (concentration0 * concentration1)
        / ((concentration1 + concentration0 + 1) * ((concentration1 + concentration0) ** 2)).to(
            original_dtype
        ),
    )


def beta_params_kl(params_p, params_q):
    """
    Compute the KL divergence between two beta distributions parameterized by params_p and params_q.
    :param params_p: Tensor of shape (batch_size, 2)
    :param params_q: Tensor of shape (batch_size, 2)
    :return: Tensor of shape (batch_size,)
    """
    original_dtype = params_p.dtype
    params_p: torch.Tensor = params_p.float()
    params_q: torch.Tensor = params_q.float()
    params_p = params_p.exp()
    params_q = params_q.exp()
    concentration1_p = params_p[:, 0]
    concentration0_p = params_p[:, 1]
    concentration1_q = params_q[:, 0]
    concentration0_q = params_q[:, 1]

    beta_distribution_p = torch.distributions.Beta(concentration1_p, concentration0_p)
    beta_distribution_q = torch.distributions.Beta(concentration1_q, concentration0_q)
    kl_divergence = torch.distributions.kl_divergence(beta_distribution_p, beta_distribution_q)
    return kl_divergence.to(original_dtype)


def bernoulli_params_kl(params_p, params_q):
    """
    Compute the KL divergence between two bernoulli distributions parameterized by params_p and params_q.
    :param params_p: Tensor of shape (batch_size, 2)
    :param params_q: Tensor of shape (batch_size, 2)
    :return: Tensor of shape (batch_size,)
    """
    original_dtype = params_p.dtype
    params_p: torch.Tensor = params_p.float()
    params_q: torch.Tensor = params_q.float()
    logprobs_p = torch.nn.functional.log_softmax(params_p, dim=-1)
    logprobs_q = torch.nn.functional.log_softmax(params_q, dim=-1)
    probs_p = logprobs_p.exp()
    kl_divergence = (probs_p * (logprobs_p - logprobs_q)).sum(dim=-1)
    return kl_divergence.to(original_dtype)


class Meme(torch.nn.Module):
    def __init__(self, lm, score_type="bernoulli", layers_to_use=(-1,), use_cls_token=False):
        """
        Initialize the ORCA model.
        :param lm: pre-trained language model (e.g., Gemma, Llama, etc.)
        :param score_type: Type of scoring function to use, either 'bernoulli' or 'beta'.
        :param layers_to_use: List of layer concatenate as input to the scorer. If None, all layers will be used.
        """
        super().__init__()
        if layers_to_use is None:
            if hasattr(lm.config, "text_config"):  # For larger Gemma models
                layers_to_use = list(range(lm.config.text_config.num_hidden_layers))
            else:
                layers_to_use = list(range(lm.config.num_hidden_layers))
        self.layers_to_use = layers_to_use
        self.lm = lm
        self.use_cls_token = use_cls_token
        if hasattr(lm.config, "text_config"):
            lm_hidden_size = lm.config.text_config.hidden_size
        else:
            lm_hidden_size = lm.config.hidden_size
        if use_cls_token:
            self.cls_token = torch.nn.Parameter(torch.randn(lm_hidden_size))
        else:
            self.cls_token = None
        self.linear = torch.nn.Linear(lm_hidden_size * len(self.layers_to_use), 2)
        self.score_type = score_type
        self.scoring_function = {
            "bernoulli": bernoulli_kl,
            "beta": beta_llh,
            "mse": mse_loss,
            "bmm": beta_moment_matching,
        }.get(score_type)
        self.param_kl_function = {
            "bernoulli": bernoulli_params_kl,
            "beta": beta_params_kl,
            "mse": beta_params_kl,
            "bmm": beta_params_kl,
        }.get(score_type)
        self.config = {
            "score_type": self.score_type,
            "layers_to_use": self.layers_to_use,
            "use_cls_token": self.use_cls_token,
        }

    def forward(self, x, prior_params=None):
        input_ids, input_lengths = x["input_ids"], x["input_len"]
        input_embeddings = self.lm.get_input_embeddings()(input_ids)
        if self.cls_token is not None:
            cls_token = self.cls_token.unsqueeze(0).expand(input_embeddings.shape[0], -1, -1)
            input_embeddings = torch.cat([input_embeddings, cls_token], dim=1)
            input_lengths += 1
        model_outputs = self.lm(
            inputs_embeds=input_embeddings,
            output_hidden_states=True,
            attention_mask=x["attention_mask"],
        )
        hidden_states = [model_outputs.hidden_states[i] for i in self.layers_to_use]
        hidden_states = [
            hd[torch.arange(hd.shape[0]), input_lengths - 1, :] for hd in hidden_states
        ]
        # hidden_states = [hd[:, -1, :] for hd in hidden_states]
        hidden_states = torch.cat(hidden_states, dim=-1)
        params = self.linear(hidden_states).squeeze(-1)

        labels = x.get("labels", None)
        score, prob1, variance = self.scoring_function(params, labels)
        score = score.mean() if score is not None else None
        if prior_params is not None:
            # prior_params = x['prior_params']
            kl_divergence = self.param_kl_function(params, prior_params)
            if score is not None:
                score = score + kl_divergence.mean()
            else:
                score = kl_divergence
        else:
            kl_divergence = None

        return {
            "params": params,
            "loss": score,  # score.sum() if score is not None else None,
            "prob1": prob1,
            "prob0": 1 - prob1,
            "variance": variance,
            "kl_divergence": kl_divergence.sum() if kl_divergence is not None else None,
        }

    def save_to_directory(self, dir_path):
        """
        Save the model to a directory.
        :param dir_path: Directory path to save the model.
        """
        os.makedirs(os.path.join(dir_path, "lm"), exist_ok=True)
        with open(os.path.join(dir_path, "config.yaml"), "w") as f:
            yaml.dump(self.config, f)
        self.lm.save_pretrained(os.path.join(dir_path, "lm"))
        state_dict_minus_lm = self.state_dict()
        for k in list(state_dict_minus_lm.keys()):
            if k.startswith("lm."):
                del state_dict_minus_lm[k]
        torch.save(state_dict_minus_lm, os.path.join(dir_path, "model_minus_lm.pt"))

    @classmethod
    def load_from_directory(cls, dir_path, lm, device=None):
        """
        Load the model from a directory.
        :param dir_path: Directory path to load the model from.
        :param device: Device to load the model on.
        :return: Loaded model instance.
        """
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, "config.yaml"), "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        model = cls(
            lm=lm,
            score_type=config["score_type"],
            layers_to_use=config["layers_to_use"],
        )
        model.lm = lm
        model.load_state_dict(
            torch.load(
                os.path.join(dir_path, "model_minus_lm.pt"), map_location=device, weights_only=True
            ),
            strict=False,
        )
        model.to(device)
        return model
