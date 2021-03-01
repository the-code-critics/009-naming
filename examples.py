def example_pandas():
    import pandas as pd

    people = pd.DataFrame({"age": [42, 12, 5, 8, 15, 65],
                           "height": [172, 155, 110, 120, 158, 168]})
    is_adult = people.age >= 18
    adult_height = people[is_adult].height.mean()
    print(adult_height)


def example_named_tensor():
    import torch as to  # requires PyTorch >= 1.7

    img = to.ones(2, 2, 3)
    print(img.sum(dim=2))

    img = to.ones(2, 2, 3, names=('H', 'W', 'C'))
    print(img.names)
    print(img.sum('C'))


def example_einops():
    import torch as to
    from einops import reduce

    img = to.ones(2, 2, 3)
    y = reduce(img, 'H W C -> H W', 'sum')
    print(y)


def example_pipeline():
    from itertools import islice

    numbers = range(10)
    gt5 = lambda x: x > 5
    take3 = lambda x: list(islice(x, 3))
    big3 = take3(filter(gt5, numbers))
    print(big3)


def example_nutsflow_pipeline():
    from nutsflow import Range, Filter, Take, Collect, _

    Range(10) >> Filter(_ > 5) >> Take(3) >> Collect()


if __name__ == '__main__':
    example_pandas()
    example_named_tensor()
    example_einops()
    example_pipeline()
    example_nutsflow_pipeline()
