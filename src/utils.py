def view_with_k(t, k):
    return t.view(t.shape[0], k, *t.shape[1:])
