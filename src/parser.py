
import json
from pydantic import BaseModel, model_validator, ConfigDict
from typing import Any
from os.path import exists
import sys


class InputsDict(BaseModel):
    """ dict verification of input json """
    model_config = ConfigDict(extra="forbid")
    prompt: str


class FuncArgs(BaseModel):
    """ dict verification of func args json """
    model_config = ConfigDict(extra="forbid")
    type: str


class ReturnArgs(BaseModel):
    """ dict verification of func return json """
    model_config = ConfigDict(extra="forbid")
    type: str


class FuncDefsDict(BaseModel):
    """ dict verification of func json """

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, FuncArgs]
    returns: ReturnArgs


class ValidDefsJson(BaseModel):
    """ verification of func def json """
    data: list[FuncDefsDict]


class ValidInputJson(BaseModel):
    """ verification of inputs json """
    data: list[InputsDict]


class ParserJson(BaseModel):
    """ Json class verification """

    path: str
    validation: type[ValidDefsJson] | type[ValidInputJson]
    filtered: list[Any] = []

    @model_validator(mode="after")
    def load_file(self) -> "ParserJson":
        """ open, read, and get json from a file """
        try:
            with open(self.path, 'r') as file:
                data: str = file.read()
                self.filtered = json.loads(data)
            self.filtered = self.validation(data=self.filtered).data
        except Exception as e:
            raise ValueError(f"JSON can't be extract : {e}")
        return self


class ParserOutput(BaseModel):
    """ output str verification """
    path: str

    @model_validator(mode="after")
    def choose_file_name(self) -> "ParserOutput":
        """ choose the file name dependig of existing files """
        i: int = 0

        if not self.path.endswith(".json"):
            raise ValueError("output must finish with .json")
        while i < 101:
            if exists(self.path):
                self.path = self.path.removesuffix(".json") + "_1.json"
            else:
                if i != 0:
                    print(
                        f"this file name already used"
                        f", name set to : {self.path}",
                        file=sys.stderr)
                break
            i += 1
        if i == 101:
            raise ValueError("too many files already created, "
                             "please make your folder clean")
        return self


class Parser(BaseModel):
    """ Main verification class """

    args: list[str]
    function_definition: list[FuncDefsDict] = []
    inputs: list[InputsDict] = []
    output: str = ""
    strict: bool = False
    _end: bool = False

    def validation(self) -> "Parser":
        """ main validation process """

        check_fdf: bool = False
        check_inp: bool = False
        check_out: bool = False
        self.args.append("")

        verified = [False for _ in range(len(self.args))]
        for i in range(1, len(self.args) - 1):
            match self.args[i]:
                case "--functions_definition":
                    check_fdf = True
                    self.function_definition.extend(ParserJson(
                            path=self.args[i + 1],
                            validation=ValidDefsJson
                            ).filtered)
                    verified[i + 1] = True
                case "--input":
                    check_inp = True
                    self.inputs.extend(ParserJson(
                            path=self.args[i + 1],
                            validation=ValidInputJson).filtered)
                    verified[i + 1] = True
                case "--output":
                    if check_out:
                        raise ValueError("only one output name can be set")
                    check_out = True
                    self.output = ParserOutput(path=self.args[i + 1]).path
                    verified[i + 1] = True
                case "--strict":
                    self.strict = True
                case _:
                    if not verified[i]:
                        raise ValueError(
                                f"unknow parameter {self.args[i]} at {i}")

        if self._end:
            return self
        self.args = [""]
        if not check_fdf:
            self.args.extend([
                "--functions_definition",
                "data/input/functions_definition.json"])
            self._end = True
        if not check_inp:
            self.args.extend([
                "--input",
                "data/input/function_calling_tests.json"])
            self._end = True
        if not check_out:
            self.args.extend(["--output", "data/output/output.json"])
            self._end = True
        if len(self.args) > 1:
            self.validation()

        return self

    @model_validator(mode="after")
    def check_args(self) -> "Parser":
        """ Pydantic validation method """
        self.validation()
        return self
