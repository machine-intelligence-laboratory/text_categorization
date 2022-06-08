from pathlib import Path

from ap.utils.general import recursively_unlink, ensure_directory, batch_names


def test_recursively_unlink():
    tmp_dir_path = Path('tests/data/my_tmp_dir')
    tmp_dir_path.mkdir()
    tmp_dir_path.joinpath('tmp_dir1').mkdir()
    tmp_dir_path.joinpath('tmp_dir2').mkdir()
    with open(tmp_dir_path.joinpath('tmp_file_1'), 'w'):
        pass
    assert tmp_dir_path.exists()
    recursively_unlink(tmp_dir_path)
    assert not tmp_dir_path.exists()


def test_ensure_directory():
    tmp_dir_path = 'tests/data/my_tmp_dir'
    assert not Path(tmp_dir_path).exists()
    ensure_directory(tmp_dir_path)
    assert Path(tmp_dir_path).exists()
    recursively_unlink(Path(tmp_dir_path))


def test_batch_names():
    num_names = 3
    batch_names_generator = batch_names('aaaaac', num_names)
    batch_names_list = list(batch_names_generator)
    assert len(batch_names_list) == num_names
    assert batch_names_list == ['aaaaad', 'aaaaae', 'aaaaaf']

    batch_names_generator = batch_names('aaaaaz', 5)
    assert list(batch_names_generator) == ['aaaaba', 'aaaabb', 'aaaabc', 'aaaabd', 'aaaabe']
