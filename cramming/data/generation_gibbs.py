""" Sampling for MLM. Implementation based on Yamakoshi et al. "Probing BERT's priors with serial reproduction chains"
"""

import torch
import torch.nn.functional as F
import time


class UniformGibbs:
    """This is code based on https://github.com/taka-yamakoshi/TelephoneGame/blob/master/model/bert.py."""

    def __init__(self, sentences, model, mask_id, device, sweep_order):
        self.sentences = sentences
        self.model = model
        self.mask_id = mask_id
        self.device = device
        self.edit_rate = torch.zeros(self.sentences.shape[0], device=self.device)
        self.accept_rate = torch.ones(self.sentences.shape[0], device=self.device)  # always accept gibbs samples
        self.sweep_order = sweep_order

    def get_rand_list(self, seq_len):
        # exclude first/last tokens (CLS/SEP) from positions
        if self.sweep_order == "ascend":
            rand_list = torch.arange(seq_len - 2, device=self.device) + 1
        elif self.sweep_order == "descend":
            rand_list = torch.arange(seq_len - 2, 0, -1, device=self.device)
        elif self.sweep_order == "random_sweep":
            rand_list = torch.randperm(seq_len - 2, device=self.device) + 1
        elif self.sweep_order == "random":
            rand_list = torch.randint(seq_len - 2, size=(seq_len - 2,), device=self.device) + 1
        else:
            print("Invalid sweep_order")
        return rand_list

    @torch.no_grad()
    def step(self, iter_num, pos):
        probs = self.mask_prob(pos, self.sentences)
        sentences, edit_loc = self.sample_words(probs, pos, self.sentences)
        return sentences, edit_loc.float()

    @torch.no_grad()
    def sweep(self, iter_num):
        seq_len = self.sentences.shape[1]
        rand_list = self.get_rand_list(seq_len)
        edit_locs = torch.zeros(size=(self.sentences.shape[0], len(rand_list)), device=self.device)
        for pos_id, pos in enumerate(rand_list):
            self.sentences, edit_locs[:, pos_id] = self.step(iter_num, pos)
        self.edit_rate = torch.mean(edit_locs, axis=1)

    def sample_words(self, probs, pos, sentences):
        chosen_words = torch.multinomial(torch.exp(probs), num_samples=1).squeeze(dim=-1)
        new_sentences = sentences.clone()
        new_sentences[:, pos] = chosen_words
        edit_loc = new_sentences[:, pos] != sentences[:, pos]
        return new_sentences, edit_loc

    def mask_prob(self, position, sentences):
        masked_sentences = sentences.clone()
        masked_sentences[:, position] = self.mask_id
        logits = self.model(masked_sentences)[0]
        return F.log_softmax(logits[:, position], dim=-1)

    @torch.no_grad()
    def get_total_score(self, sentences):
        sent_probs = torch.zeros_like(sentences).float()
        # Calculate masked probabilities for the actual words in the sentences
        for j in range(1, sentences.shape[1] - 1):
            probs = self.mask_prob(j, sentences)
            for i in range(sentences.shape[0]):
                # Look up probability of the actual word at this position
                sent_probs[i, j] = probs[i, sentences[i, j]]
        # Sum log probs for each sentence in batch
        return torch.sum(sent_probs, axis=1)


@torch.no_grad()
def run_chains(model, tokenizer, sampling_method="gibbs_mixture_mask_init", device=torch.device("cpu")):

    cls_token_id = tokenizer.convert_tokens_to_ids("<cls>")
    sep_token_id = tokenizer.convert_tokens_to_ids("<sep>")
    mask_token_id = tokenizer.convert_tokens_to_ids("<mask>")
    seq_length = tokenizer.model_max_length
    sweep_order = "random"
    batch_size = 32
    chain_len = 51000
    burn_in_period = 1000
    print_every_iter = 100
    epsilon = 1e-3

    mask_input = (
        torch.tensor([cls_token_id] + [mask_token_id for _ in range(seq_length - 2)] + [sep_token_id])
        .to(device)
        .expand((batch_size, -1))  # This is approximately ok, but not exactly what the model is trained on for us
    )

    for i in range(batch_size):
        print(f"Beginning batch {i}")
        start = time.time()

        # set up the input
        init_input = mask_input.clone()
        # set up the sampler
        sampler = UniformGibbs(init_input, model, mask_token_id, device, sweep_order)

        switch = 0
        num_switches = 0
        for iter_num in range(chain_len):
            # write out all steps for iter_num<100
            if iter_num < burn_in_period:
                seq_len = sampler.sentences.shape[1]
                # exclude first/last tokens (CLS/SEP) from positions
                rand_list = sampler.get_rand_list(seq_len)
                for pos_id, pos in enumerate(rand_list):
                    sampler.sentences, sampler.edit_rate = sampler.step(iter_num, pos)

            else:
                if iter_num % print_every_iter == 0:
                    print(
                        f"It {iter_num}. Score: {sampler.get_total_score(sampler.sentences).cpu().detach().numpy()}"
                        f"Edit rate: { sampler.edit_rate}. Step: {sampler.sentences.shape[1]-3}. "
                        f"Switch: {switch} Accept: {sampler.accept_rate}"
                    )
                    for sentence in sampler.sentences:
                        print(tokenizer.decode(sentence))
                    switch = 0
                sampler.sweep(iter_num)
                if "mixture" in sampling_method and iter_num >= 1000 and torch.rand(1) < epsilon:
                    switch = 1
                    num_switches += 1
                    new_init_input = mask_input.clone()
                    sampler = UniformGibbs(new_init_input, model, mask_token_id, device, sweep_order)
                    for _ in range(100):
                        sampler.sweep(iter_num)
        print(f"# of switches: {num_switches}")
        print(f"Time it took for {i}th batch: {time.time()-start}")


def test_generation():
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

    run_chains(model, tokenizer, sampling_method="gibbs_mixture_mask_init", device=torch.device("cpu"))


# Test:
if __name__ == "__main__":
    test_generation()
