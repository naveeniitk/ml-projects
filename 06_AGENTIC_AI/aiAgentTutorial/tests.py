import unittest
from functions.get_files_info import get_files_info
from functions.get_files_contents import get_files_contents


class TestGetFilesInfo(unittest.TestCase):

    def setUp(self):
        self.working_directory = "calculator"

    def test_root_contents(self):
        result = get_files_info(self.working_directory)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_current_directory(self):
        result = get_files_info(self.working_directory, ".")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_parent_directory(self):
        result = get_files_info(self.working_directory, "../")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_bin_directory(self):
        result = get_files_info(self.working_directory, "/bin")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_pkg_directory(self):
        result = get_files_info(self.working_directory, "pkg")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class TestGetFilesContents(unittest.TestCase):

    def setUp(self):
        self.working_directory = "calculator"

    def test_read_lorem_file(self):
        result = get_files_contents(self.working_directory, "lorem.txt")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_read_main_file(self):
        result = get_files_contents(self.working_directory, "main.py")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_read_functions_file(self):
        result = get_files_contents(
            self.working_directory, "functions/get_files_info.py"
        )
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_read_readme_file(self):
        result = get_files_contents(self.working_directory, "readme/README.md")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
