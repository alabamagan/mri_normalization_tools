__all__ = ['repeat_zip']

def repeat_zip(*args):
    r"""
    Zip with shorter columns repeated until the longest column is fully iterated.

    Args:
        *args (list of iterables)

    Examples:
        >>> x = [tuple([1]), ('a', 'b'), ('Z', 'D', 'E', 'F')]
        >>> for row in repeat_zip(*x):
        >>>    print(row)
        >>> #(1, 'a', 'Z')
        >>> #(1, 'b', 'D')
        >>> #(1, 'a', 'E')
        >>> #(1, 'b', 'F')
    """
    iterators = [iter(it) for it in args]
    finished = {i: False for i in range(len(iterators))}
    while True:
        values = []
        for i, it in enumerate(iterators):
            try:
                value = next(it)
            except StopIteration:
                finished[i] = True
                iterators[i] = iter(args[i])
                value = next(iterators[i])
            values.append(value)
            if all([x[1] for x in finished.items()]):
                return
        yield tuple(values)