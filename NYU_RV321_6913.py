import argparse
import os

from colorama import Fore, Back, Style
from signal import signal, SIGINT
from sys import exit

import abc

from abc import ABC

import copy

from riscvmodel.isa import Instruction
from riscvmodel.code import decode, MachineDecodeError
from bitstring import BitArray

MemSize = 1000 # memory size, in reality, the memory size should be 2^32, but for this lab, for the space resaon, we keep it as this large number, but the memory is still 32-bit addressable.

class InsMem(object):

    def __init__(self, name, ioDir):
        self.id = name

        with open(ioDir + "/imem.txt") as im:
            self.IMem = [data.replace("\n", "") for data in im.readlines()]

    def read_instr(self, read_address):
        # DONE: Handle word addressing - use nearest lower multiple for 4 for address = x - x % 4
        read_address = read_address - read_address % 4
        if len(self.IMem) < read_address + 4:
            raise Exception("Instruction MEM - Out of bound access")
        return "".join(self.IMem[read_address: read_address + 4])


class DataMem(object):
    def __init__(self, name, ioDir):
        self.id = name
        self.ioDir = ioDir
        with open(ioDir + "\\dmem.txt") as dm:
            self.DMem = [data.replace("\n", "") for data in dm.readlines()]

    def read_data(self, read_address):
        # read data memory
        # return 32-bit signed int value

        #Handle word addressing - use nearest lower multiple for 4 for address = x - x % 4
        read_address = read_address - read_address % 4
        if len(self.DMem) < read_address + 4:
            raise Exception("Data MEM - Out of bound access")
        return BitArray(bin="".join(self.DMem[read_address: read_address + 4])).int32

    def write_data_mem(self, address, write_data):
        # write data into byte addressable memory
        # Assuming data as 32 bit signed integer

        # Converting from int to bin

        # Handle word addressing - use nearest lower multiple for 4 for address = x - x % 4
        address = address - address % 4
        write_data = '{:032b}'.format(write_data & 0xffffffff)

        left, right, zeroes = [], [], []

        if address <= len(self.DMem):
            left = self.DMem[:address]
        else:
            left = self.DMem
            zeroes = ["0" * 8] * (address - len(self.DMem))
        if address + 4 <= len(self.DMem):
            right = self.DMem[address + 4:]

        self.DMem = left + zeroes + [write_data[i: i + 8] for i in range(0, 32, 8)] + right

    def output_data_mem(self):
        res_path = self.ioDir + "/" + self.id + "_DMEMResult.txt"
        with open(res_path, "w") as rp:
            rp.writelines([str(data) + "\n" for data in self.DMem])
            rp.writelines([ '0'*8 + '\n' for _ in range(MemSize - len(self.DMem))])

class RegisterFile(object):
    def __init__(self, ioDir):
        self.outputFile = ioDir + "RFResult.txt"
        self.Registers = [0x0 for i in range(32)]

    def read_rf(self, reg_addr):
        return self.Registers[reg_addr]

    def write_rf(self, reg_addr, wrt_reg_data):
        if reg_addr != 0:
            self.Registers[reg_addr] = wrt_reg_data

    def output_rf(self, cycle):
        op = ["State of RF after executing cycle:\t" + str(cycle) + "\n"]
        op.extend(['{:032b}'.format(val & 0xffffffff) + "\n" for val in self.Registers])
        if cycle == 0:perm = "w"
        else: perm = "a"
        with open(self.outputFile, perm) as file:
            file.writelines(op)

class State(object):
    def __init__(self):
        self.IF = {"nop": False, "PC":0, "halt": False}
        self.ID = {"nop": False, "Instr": 0, "halt": False}
        self.EX = {"nop": False, "Read_data1": 0, "Read_data2": 0, "Rd":0, "Store_data":0, "Imm": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, "is_I_type": False, "rd_mem": False, 
                   "wrt_mem": False, "opcode": 0, "wrt_enable": False, "instruction_ob": None, "halt": False}
        self.MEM = {"nop": False, "ALUresult": 0, "Store_data": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, "rd_mem": False, 
                   "wrt_mem": False, "wrt_enable": False, "instruction_ob": None, "halt": False}
        self.WB = {"nop": False, "Wrt_data": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, "wrt_enable": False, "instruction_ob": None, "halt": False}

    def nop_init(self):
        self.IF["nop"] = False
        self.ID["nop"] = True
        self.EX["nop"] = True
        self.MEM["nop"] = True
        self.WB["nop"] = True

    def __str__(self):
        # DONE: update __str__ to make use of individual State objects
        return "\n\n".join([str(self.IF), str(self.ID), str(self.EX), str(self.MEM), str(self.WB)])

class InstructionBase(metaclass=abc.ABCMeta):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        self.instruction = instruction
        self.memory = memory
        self.registers = registers
        self.state = state
        self.nextState = nextState
        self.stages = self.memory.id

    def decode_ss(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def execute_ss(self, *args, **kwargs):
        pass

    def mem_ss(self, *args, **kwargs):
        pass

    def wb_ss(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def decode_fs(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def execute_fs(self, *args, **kwargs):
        pass

    def mem_fs(self, *args, **kwargs):

        # print(self.state.MEM)
        wb_state = State().WB
        wb_state.update({
            'instruction_ob':self.state.MEM['instruction_ob'],
            'nop':self.state.MEM['nop'],
            'Wrt_data':self.state.MEM['Store_data'],
            'Wrt_reg_addr':self.state.MEM['Wrt_reg_addr'],
            'wrt_enable':self.state.MEM['wrt_enable'],
            'halt':self.state.MEM['halt']
        }
        )
        # print(wb_state)
        self.nextState.WB = wb_state

    def wb_fs(self, *args, **kwargs):
        if self.state.WB['wrt_enable']:
            self.registers.write_rf(self.state.WB['Wrt_reg_addr'], self.state.WB['Wrt_data'])

    def decode(self, *args, **kwargs):
        if self.stages == "SS":
            return self.decode_ss(*args, **kwargs)
        else:
            self.state = kwargs["state"]
            self.nextState = kwargs["nextState"]
            self.memory = kwargs["memory"]
            self.registers = kwargs["registers"]
            return self.state, self.nextState, self.memory, self.registers, self.decode_fs(*args, **kwargs)

    def execute(self, *args, **kwargs):
        if self.stages == "SS":
            return self.execute_ss(*args, **kwargs)
        else:
            self.state = kwargs["state"]
            self.nextState = kwargs["nextState"]
            self.memory = kwargs["memory"]
            self.registers = kwargs["registers"]
            response = self.execute_fs(*args, **kwargs)
            return self.state, self.nextState, self.memory, self.registers, response

    def mem(self, *args, **kwargs):
        if self.stages == "SS":
            return self.mem_ss(*args, **kwargs)
        else:
            self.state = kwargs["state"]
            self.nextState = kwargs["nextState"]
            self.memory = kwargs["memory"]
            self.registers = kwargs["registers"]
            response = self.mem_fs(*args, **kwargs)
            return self.state, self.nextState, self.memory, self.registers, response

    def wb(self, *args, **kwargs):
        if self.stages == "SS":
            return self.wb_ss(*args, **kwargs)
        else:
            self.state = kwargs["state"]
            self.nextState = kwargs["nextState"]
            self.memory = kwargs["memory"]
            self.registers = kwargs["registers"]
            response = self.wb_fs(*args, **kwargs)
            return self.state, self.nextState, self.memory, self.registers, response


class InstructionRBase(InstructionBase, ABC):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(InstructionRBase, self).__init__(instruction, memory, registers, state, nextState)
        self.rs1 = instruction.rs1
        self.rs2 = instruction.rs2
        self.rd = instruction.rd

    def wb_ss(self, *args, **kwargs):
        data = kwargs['alu_result']
        return self.registers.write_rf(self.rd, data)

    def decode_fs(self, *args, **kwargs):
        ex_state = State().EX

        # TODO: Handle Hazards
        #   set nop for EX state
        #   will be applicable in R, I, S, B, J type instructions

        ex_state.update({
            'instruction_ob':self,
            'nop' :self.state.ID['nop'],
            'Read_data1' :self.registers.read_rf(self.rs1),
            'Read_data2' :self.registers.read_rf(self.rs2),
            'Rd' :self.rd,
            'wrt_enable' :True
        })

        # Stall
        if self.state.EX['Rd'] in [self.rs1, self.rs2] \
                and self.state.EX['rd_mem'] \
                and  self.rs1 != 0 and self.rs2 != 0:
            print("Stall")
            print(ex_state)
            ex_state['nop'] = True
            self.state.IF['PC'] -= 4
            self.nextState.EX = ex_state
            # self.nextState.IF['instruction_count'] = self.nextState.IF['instruction_count'] - 1
            return

        # Forwarding
        if self.state.MEM['rd_mem'] and self.state.MEM['wrt_enable'] and \
            not self.state.MEM['wrt_mem'] and \
            self.state.MEM['Wrt_reg_addr'] == self.rs1 and self.rs1 != 0:
            print("Forwarding")
            print(ex_state)
            ex_state['Read_data1'] = self.nextState.WB['Wrt_data']

        if self.state.MEM['rd_mem'] and self.state.MEM['wrt_enable'] and \
            not self.state.MEM['wrt_mem'] and \
            self.state.MEM['Wrt_reg_addr'] == self.rs2 and self.rs2 != 0:
            print("Forwarding")
            print(ex_state)
            ex_state['Read_data2'] = self.nextState.WB['Wrt_data']

        if not self.state.EX['rd_mem'] and self.state.EX['wrt_enable'] and \
            not self.state.EX['wrt_mem'] and self.state.EX['Rd'] == self.rs1 and \
            self.rs1 != 0:
            print("Forwarding")
            print(ex_state)

            ex_state['Read_data1'] = self.nextState.MEM['Store_data']

        if not self.state.EX['rd_mem'] and self.state.EX['wrt_enable'] and \
            not self.state.EX['wrt_mem'] and self.state.EX['Rd'] == self.rs2 and \
            self.rs2 != 0:
            print("Forwarding")
            print(ex_state)

            ex_state['Read_data2'] = self.nextState.MEM['Store_data']

        self.nextState.EX = ex_state

    def execute_fs(self, *args, **kwargs):
        mem_state = State().MEM
        mem_state.update({
            'instruction_ob':self,
            'nop' :self.state.EX['nop'],
            'Wrt_reg_addr' :self.state.EX['Rd'],
            'wrt_enable' :True,
            'halt' :self.state.EX['halt']
        })
        self.nextState.MEM = mem_state


class InstructionIBase(InstructionBase, ABC):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(InstructionIBase, self).__init__(instruction, memory, registers, state, nextState)
        self.rs1 = instruction.rs1
        self.rd = instruction.rd
        self.imm = instruction.imm.value
        # print(self.imm)
    def wb_ss(self, *args, **kwargs):
        data = kwargs['alu_result']
        return self.registers.write_rf(self.rd, data)

    def decode_fs(self, *args, **kwargs):
        ex_state = State().EX
        ex_state.update({
            'instruction_ob':self,
            'nop' :self.state.ID['nop'],
            'Read_data1' :self.registers.read_rf(self.rs1),
            'Read_data2' :self.imm,
            'Rd' :self.rd,
            'wrt_enable' :True,
            'halt' :self.state.ID['halt']
        })


        # Stall
        if self.state.EX['Rd'] == self.rs1 and self.state.EX['rd_mem'] and self.rs1 != 0:
            print("Stall")
            print(ex_state)
            ex_state['nop'] = True
            self.state.IF['PC'] -= 4
            self.nextState.EX = ex_state
            # self.nextState.IF['instruction_count'] = self.nextState.IF['instruction_count'] - 1
            return

        # Forwarding
        if self.state.MEM['rd_mem'] and self.state.MEM['wrt_enable'] and \
            not self.state.MEM['wrt_mem'] and self.state.MEM['Wrt_reg_addr'] == self.rs1 \
            and self.rs1 != 0:
            print("Forwarding")
            print(ex_state)

            ex_state['Read_data1'] = self.nextState.WB['Wrt_data']

        if not self.state.EX['rd_mem'] and self.state.EX['wrt_enable'] and \
            not self.state.EX['wrt_mem'] and self.state.EX['Rd'] == self.rs1 and \
            self.rs1 != 0:
            print("Forwarding")
            print(ex_state)

            ex_state["Read_data1"] = self.nextState.MEM['Store_data']

        self.nextState.EX = ex_state

    def execute_fs(self, *args, **kwargs):
        # mem_state = MEMState()
        mem_state = State().MEM

        mem_state.update({
            'instruction_ob':self,
            'nop' :self.state.EX['nop'],
            'Wrt_reg_addr' :self.state.EX['Rd'],
            'wrt_enable' :True,
            'halt' :self.state.EX['halt']
        })
        self.nextState.MEM = mem_state


class InstructionSBase(InstructionBase, ABC):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(InstructionSBase, self).__init__(instruction, memory, registers, state, nextState)
        self.rs1 = instruction.rs1
        self.rs2 = instruction.rs2
        self.imm = instruction.imm.value

    def mem_ss(self, *args, **kwargs):
        address = kwargs['alu_result']
        data = self.registers.read_rf(self.rs2)
        self.memory.write_data_mem(address, data)

    def decode_fs(self, *args, **kwargs):
        # ex_state = EXState()
        ex_state = State().EX

        ex_state.update({
            'instruction_ob':self,
            'nop' :self.state.ID['nop'],
            'Read_data1' :self.registers.read_rf(self.rs1),
            'Read_data2' :self.imm,
            'Store_data' :self.registers.read_rf(self.rs2),
            'Rd' :self.rs2,
            'wrt_mem' :True,
            'halt':self.state.ID['halt'] 
        })
        # Stall
        if self.state.EX['Rd'] in [self.rs1, self.rs2] and \
            self.state.EX['rd_mem'] and self.rs1 != 0 and self.rs2 != 0:

            print("Stall at 393")
            ex_state['nop'] = True
            self.state.IF['PC'] -= 4
            self.nextState.EX = ex_state
            # self.nextState.IF['instruction_count'] = self.nextState.IF['instruction_count'] - 1
            return

        # Forwarding
        if not self.state.EX['rd_mem'] and self.state.EX['wrt_enable'] and \
            not self.state.EX['wrt_mem'] and self.state.EX['Rd'] == self.rs1 \
            and self.rs1 != 0:
            print("Forwarding")
            print(ex_state)

            ex_state['Read_data1'] = self.nextState.MEM['Store_data']

        if not self.state.EX['rd_mem'] and self.state.EX['wrt_enable'] and \
            not self.state.EX['wrt_mem'] and self.state.EX['Rd'] == self.rs2 and \
            self.rs2 != 0:
            print("Forwarding")
            print(ex_state)

            ex_state['Store_data'] = self.nextState.MEM['Store_data']

        if not self.state.MEM['rd_mem'] and self.state.MEM['wrt_enable'] and \
            not self.state.MEM['wrt_mem'] and self.state.MEM['Wrt_reg_addr'] == self.rs1 and \
            self.rs1 != 0:
            print("Forwarding")
            print(ex_state)

            ex_state['Read_data1'] = self.nextState.WB['Wrt_data']

        if not self.state.MEM['rd_mem'] and self.state.MEM['wrt_enable'] and \
            not self.state.MEM['wrt_mem'] and self.state.MEM['Wrt_reg_addr'] == self.rs2 and \
            self.rs2 != 0:
            print("Forwarding")
            print(ex_state)

            ex_state['Store_data'] = self.nextState.WB['Wrt_data']

        self.nextState.EX = ex_state

    def execute_fs(self, *args, **kwargs):
        # mem_state = MEMState()
        mem_state = State().MEM
        mem_state.update({
            'instruction_ob':self,
            'nop' :self.state.EX['nop'],
            'data_address' :self.state.EX['Read_data1'] + self.state.EX['Read_data2'],
            'Store_data' :self.state.EX['Store_data'],
            'wrt_mem' :True,
            'halt':self.state.ID['halt']
        })
        self.nextState.MEM = mem_state

    def mem_fs(self, *args, **kwargs):
        if self.state.MEM['wrt_mem']:
            self.memory.write_data_mem(self.state.MEM['data_address'], \
                                       self.state.MEM['Store_data'])
        # wb_state = WBState()
        wb_state = State().WB
        wb_state.update({
            'instruction_ob':self
        })

        print(wb_state)
        self.nextState.WB = wb_state


class InstructionBBase(InstructionBase, ABC):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(InstructionBBase, self).__init__(instruction, memory, registers, state, nextState)
        self.rs1 = instruction.rs1
        self.rs2 = instruction.rs2
        self.imm = instruction.imm.value

    @abc.abstractmethod
    def take_branch(self, operand1, operand2):
        pass

    def execute_ss(self, *args, **kwargs):
        pass

    def execute_fs(self, *args, **kwargs):
        mem_state = State().MEM
        mem_state['instruction_ob'] = self
        mem_state['nop'] = True
        self.nextState.MEM = mem_state

    def decode_fs(self, *args, **kwargs):

        operand1 = self.registers.read_rf(self.rs1)
        operand2 = self.registers.read_rf(self.rs2)

        if self.state.EX['wrt_enable'] and self.state.EX['Rd'] != 0 and \
            self.state.EX['Rd'] == self.rs1 and self.rs1 != 0:
            operand1 = self.nextState.MEM['Store_data']

        if self.state.EX['wrt_enable'] and self.state.EX['Rd'] != 0 and \
            self.state.EX['Rd'] == self.rs2 and self.rs2 != 0:
            operand2 = self.nextState.MEM['Store_data']

        if self.state.MEM['wrt_enable'] and self.state.MEM['Wrt_reg_addr'] != 0 and \
            not ( self.state.EX['wrt_enable'] and self.state.EX['Rd'] != 0 and \
                self.state.EX['Rd'] == self.rs1) and \
                self.state.MEM['Wrt_reg_addr'] == self.rs1 and self.rs1 != 0:
            operand1 = self.nextState.WB['Wrt_data']

        if self.state.MEM['wrt_enable'] and self.state.MEM['Wrt_reg_addr'] != 0 and \
            not ( self.state.EX['wrt_enable'] and self.state.EX['Rd'] != 0 and \
                self.state.EX['Rd'] == self.rs2) and \
                self.state.MEM['Wrt_reg_addr'] == self.rs2 and self.rs2 != 0:
            operand2 = self.nextState.WB['Wrt_data']

        ex_state = State().EX
        ex_state['instruction_ob'] = self

        if self.take_branch(operand1, operand2):
            self.nextState.IF['PC'] = self.state.IF['PC'] + self.imm - 4
            self.nextState.ID['nop'] = True
            self.state.IF['nop'] = True
        ex_state['nop'] = True

        self.nextState.EX = ex_state


class InstructionJBase(InstructionBase, ABC):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(InstructionJBase, self).__init__(instruction, memory, registers, state, nextState)
        self.rd = instruction.rd
        self.imm = instruction.imm.value

    def execute_ss(self, *args, **kwargs):
        pass

    def decode_fs(self, *args, **kwargs):
        # ex_state = EXState()
        ex_state = State().EX
        ex_state.update({
            'instruction_ob' :self,
            'Store_data' :self.state.IF['PC'],
            'Rd' :self.rd,
            'wrt_enable' :True
        })

        print(f"EX STATE:",ex_state)
        self.nextState.IF['PC'] = self.state.IF['PC'] + self.imm - 4
        self.nextState.ID['nop'] = True
        print(f"NEXT State", self.nextState)
        self.state.IF['nop'] = True

        self.nextState.EX = ex_state

    def execute_fs(self, *args, **kwargs):
        # mem_state = MEMState()
        mem_state = State().MEM
        print(f"Before: {mem_state}")
        mem_state.update({
            'instruction_ob':self,
            'Store_data' :self.state.EX['Store_data'],
            'Wrt_reg_addr' :self.rd,
            'wrt_enable' :True
        })

        self.nextState.MEM = mem_state
        print(f"After: {self.nextState.MEM}")


class ADD(InstructionRBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(ADD, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) + self.registers.read_rf(self.rs2)

    def execute_fs(self, *args, **kwargs):
        super(ADD, self).execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] + self.state.EX['Read_data2']


class SUB(InstructionRBase):

    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(SUB, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) - self.registers.read_rf(self.rs2)

    def execute_fs(self, *args, **kwargs):
        super(SUB, self).execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] - self.state.EX['Read_data2']


class XOR(InstructionRBase):

    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(XOR, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) ^ self.registers.read_rf(self.rs2)

    def execute_fs(self, *args, **kwargs):
        super(XOR, self).execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] ^ self.state.EX['Read_data2']


class OR(InstructionRBase):

    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(OR, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) | self.registers.read_rf(self.rs2)

    def execute_fs(self, *args, **kwargs):
        super(OR, self).execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] | self.state.EX['Read_data2']


class AND(InstructionRBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(AND, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) & self.registers.read_rf(self.rs2)

    def execute_fs(self, *args, **kwargs):
        super(AND, self).execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] & self.state.EX['Read_data2']


class ADDI(InstructionIBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(ADDI, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) + self.imm

    def execute_fs(self, *args, **kwargs):
        super(ADDI, self).execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] + self.state.EX['Read_data2']


class XORI(InstructionIBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(XORI, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) ^ self.imm

    def execute_fs(self, *args, **kwargs):
        super(XORI, self).execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] ^ self.state.EX['Read_data2']


class ORI(InstructionIBase):

    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(ORI, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) | self.imm

    def execute_fs(self, *args, **kwargs):
        super(ORI, self).execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] | self.state.EX['Read_data2']


class ANDI(InstructionIBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(ANDI, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) & self.imm

    def execute_fs(self, *args, **kwargs):
        super(ANDI, self).execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] & self.state.EX['Read_data2']


class LW(InstructionIBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(LW, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) + self.imm

    def mem_ss(self, *args, **kwargs):
        address = kwargs['alu_result']
        return self.memory.read_data(address)

    def wb_ss(self, *args, **kwargs):
        data = kwargs['mem_result']
        return self.registers.write_rf(self.rd, data)

    def decode_fs(self, *args, **kwargs):
        super(LW, self).decode_fs()
        self.nextState.EX['rd_mem'] = True

    def execute_fs(self, *args, **kwargs):
        super(LW, self).execute_fs()
        self.nextState.MEM.update({
            'data_address': self.state.EX['Read_data1'] + self.state.EX['Read_data2'],
            'rd_mem' : True
        })

    def mem_fs(self, *args, **kwargs):
        super(LW, self).mem_fs(*args, **kwargs)
        if self.state.MEM['rd_mem']:
            self.nextState.WB['Wrt_data'] = self.memory.read_data(
                self.state.MEM['data_address']
            )


class SW(InstructionSBase):
    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(SW, self).__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) + self.imm


class BEQ(InstructionBBase):

    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(BEQ, self).__init__(instruction, memory, registers, state, nextState)

    def take_branch(self, operand1, operand2):
        return operand1 == operand2


class BNE(InstructionBBase):

    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(BNE, self).__init__(instruction, memory, registers, state, nextState)

    def take_branch(self, operand1, operand2):
        return operand1 != operand2


class JAL(InstructionJBase):

    def __init__(self, instruction: Instruction, memory: DataMem, registers: RegisterFile, state: State,
                 nextState: State):
        super(JAL, self).__init__(instruction, memory, registers, state, nextState)


class ADDERBTYPE:
    def __init__(self, instruction: Instruction, state: State(), registers: RegisterFile):
        self.instruction = instruction
        self.state = state
        self.registers = registers
        self.rs1 = instruction.rs1
        self.rs2 = instruction.rs2
        self.imm = instruction.imm.value

    def get_pc(self, *args, **kwargs):
        if self.instruction.mnemonic == 'beq':
            if self.registers.read_rf(self.rs1) == self.registers.read_rf(self.rs2):
                return self.state.IF['PC'] + self.imm
            else:
                return self.state.IF['PC'] + 4
        else:
            if self.registers.read_rf(self.rs1) != self.registers.read_rf(self.rs2):
                return self.state.IF['PC'] + self.imm
            else:
                return self.state.IF['PC'] + 4


class ADDERJTYPE:
    def __init__(self, instruction: Instruction, state: State(), registers: RegisterFile):
        self.instruction = instruction
        self.state = state
        self.registers = registers
        self.rd = instruction.rd
        self.imm = instruction.imm.value

    def get_pc(self, *args, **kwargs):
        self.registers.write_rf(self.rd, self.state.IF['PC'] + 4)
        return self.state.IF['PC'] + self.imm




def get_instruction_class(mnemonic):
    try:
        if mnemonic == "lb":
            mnemonic = "lw"
        
        cls = eval(mnemonic.upper())
        return cls
    except AttributeError as e:
        raise Exception("Invalid Instruction")

class Core(object):
    def __init__(self, ioDir: str, imem: InsMem, dmem: DataMem):
        self.myRF = RegisterFile(ioDir)
        self.cycle = 0
        self.halted = False
        self.ioDir = ioDir
        self.state = State()
        self.state.nop_init()
        self.nextState = State()
        self.nextState.nop_init()
        self.ext_imem: InsMem = imem
        self.ext_dmem: DataMem = dmem

    def calculate_performance_metrics(self):
        cpi = float(self.cycle) / len(self.ext_imem.IMem)
        ipc = 1 / cpi

        result_format = f"{'-'*15} Core Performance Metrics {'-'*15}\n" \
                        f"Number of cycles taken: {self.cycle}\n" \
                        f"Cycles per instruction: {cpi}\n" \
                        f"Instructions per cycle: {ipc}\n"

        print(self.ioDir[:-3] + "PerformanceMetrics_Result.txt")
        with open(self.ioDir[:-3] + "PerformanceMetrics_Result.txt", 'w') as file:
            file.write(result_format)

class SingleStageCore(Core):
    def __init__(self, ioDir: str, imem: InsMem, dmem: DataMem):
        super(SingleStageCore, self).__init__(ioDir + "/SS_", imem, dmem)
        self.opFilePath = ioDir + "/StateResult_SS.txt"
        self.stages = "Single Stage"


    def step(self):
        # IF
        instruction_bytes = self.ext_imem.read_instr(self.state.IF['PC'])
        # self.nextState.IF['instruction_count'] = self.nextState.IF['instruction_count'] + 1

        if instruction_bytes == "1" * 32:
            self.nextState.IF['nop'] = True
        else:
            self.nextState.IF['PC'] += 4

        try:
            # ID
            instruction: Instruction = decode(int(instruction_bytes, 2))

            if instruction.mnemonic in ['beq', 'bne']:
                self.nextState.IF['PC'] = ADDERBTYPE(instruction, self.state, self.myRF).get_pc()
            elif instruction.mnemonic == 'jal':
                self.nextState.IF['PC'] = ADDERJTYPE(instruction, self.state, self.myRF).get_pc()
            else:
                instruction_ob: InstructionBase = get_instruction_class(instruction.mnemonic)(instruction,
                                                                                              self.ext_dmem, self.myRF,
                                                                                              self.state,
                                                                                              self.nextState)
                # Ex
                alu_result = instruction_ob.execute()
                # Load/Store (MEM)
                mem_result = instruction_ob.mem(alu_result=alu_result)
                # WB
                wb_result = instruction_ob.wb(mem_result=mem_result, alu_result=alu_result)
        except MachineDecodeError as e:
            if "{:08x}".format(e.word) == 'ffffffff':
                pass
            else:
                raise Exception("Invalid Instruction to Decode")
        # self.halted = True
        if self.state.IF['nop']:
            self.halted = True

        self.myRF.output_rf(self.cycle)  # dump RF
        self.printState(self.nextState, self.cycle)  # print states after executing cycle 0, cycle 1, cycle 2 ...

        # The end of the cycle and updates the current state with the values calculated in this cycle
        self.state = copy.deepcopy(self.nextState)
        self.cycle += 1

    def printState(self, state, cycle):
        printstate =  ["-" * 70 + "\n", "State after executing cycle: " + str(cycle) + "\n"]

        printstate.append("IF.PC: " + str(state.IF['PC']) + "\n")
        printstate.append("IF.nop: " + str(state.IF['nop']) + "\n")

        if (cycle == 0):perm = "w"
        else: perm = "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)

class FiveStageCore(Core):
    def __init__(self, ioDir, imem, dmem):
        super(FiveStageCore, self).__init__(ioDir + "/FS_", imem, dmem)
        self.opFilePath = ioDir + "/StateResult_FS.txt"
        self.stages = "Five Stage"

    def print_current_instruction(self, cycle, stage, instruction):
        if issubclass(type(instruction), Instruction):
            print(f"{cycle}\t{stage}\t{instruction}")
        else:
            if all([x in ["0", "1"] for x in instruction]):
                try:
                    print(f"{cycle}\t{stage}\t{decode(int(instruction, 2))}")
                except MachineDecodeError as e:
                    print(f"{cycle}\t{stage}\tHalt")
            else:
                print(f"{cycle}\t{stage}\t{instruction}")


    def step(self):
        # Your implementation
        # --------------------- WB stage ----------------------
        if not self.state.WB["nop"]:
            self.print_current_instruction(self.cycle, "WB", self.state.WB["instruction_ob"].instruction)

            self.state, self.nextState, self.ext_dmem, self.myRF, _ = self.state.WB["instruction_ob"].wb(
                state=self.state,
                nextState=self.nextState,
                registers=self.myRF,
                memory=self.ext_dmem)
        else:
            self.print_current_instruction(self.cycle, "WB", "nop")


        # --------------------- MEM stage ---------------------
        if not self.state.MEM["nop"]:
            self.print_current_instruction(self.cycle, "MEM", self.state.MEM["instruction_ob"].instruction)

            self.state, self.nextState, self.ext_dmem, self.myRF, _ = self.state.MEM["instruction_ob"].mem(
                state=self.state,
                nextState=self.nextState,
                registers=self.myRF,
                memory=self.ext_dmem)
        else:

            self.nextState.WB["nop"] = True
            self.print_current_instruction(self.cycle, "MEM", "nop")


        # --------------------- EX stage ----------------------
        if not self.state.EX["nop"]:
            self.print_current_instruction(self.cycle, "EX", self.state.EX["instruction_ob"].instruction)
            self.state, self.nextState, self.ext_dmem, self.myRF, _ = self.state.EX["instruction_ob"].execute(
                state=self.state, nextState=self.nextState, registers=self.myRF, memory=self.ext_dmem)
        else:
            self.nextState.MEM["nop"] = True
            self.print_current_instruction(self.cycle, "EX", "nop")

        # --------------------- ID stage ----------------------
        if not self.state.ID["nop"]:
            self.print_current_instruction(self.cycle, "ID", self.state.ID['Instr'])

            # print("Instruction bytes", self.state.ID['Instr'])
            try:
                instruction = decode(int(self.state.ID['Instr'], 2))
                # print(instruction)
                #returns Instruction Class which can access attributes of the instruction base class
                instruction_ob  = get_instruction_class(instruction.mnemonic)(instruction,
                                                                                self.ext_dmem,
                                                                                self.myRF,
                                                                                self.state,
                                                                                self.nextState)
            
                # print(Fore.RED + f"Current State: {self.state}")
                # print(Fore.GREEN + f"Next State: {self.nextState}")



                self.state, self.nextState, self.ext_dmem, self.myRF, _ = instruction_ob.decode(state=self.state,
                                                                                                nextState=self.nextState,
                                                                                                registers=self.myRF,
                                                                                                memory=self.ext_dmem)
                # print(Fore.RED + Back.YELLOW + f"Current State: {self.state}")
                # print(Fore.GREEN + Back.YELLOW+ f"Next State: {self.nextState}")


                # print(self.state, self.nextState, self.ext_dmem, self.myRF)
            except MachineDecodeError as e:
                if "{:08x}".format(e.word) == 'ffffffff':
                    self.nextState.ID["halt"] = True
                else:
                    raise Exception("Invalid Instruction to Decode")
        else:
            self.nextState.EX["nop"] = True
            self.print_current_instruction(self.cycle, "ID", "nop")

        # --------------------- IF stage ----------------------
        if not self.state.IF["nop"]:
            self.nextState.ID['Instr'] = self.ext_imem.read_instr(self.state.IF["PC"])
            self.nextState.ID["nop"] = False
            if self.nextState.ID['Instr'] == "1" * 32:
                self.nextState.ID["nop"] = True
                self.nextState.IF["nop"] = True
            else:
                self.nextState.IF["PC"] = self.state.IF["PC"] + 4
                # self.nextState.IF['instruction_count'] = self.nextState.IF['instruction_count'] + 1
            self.print_current_instruction(self.cycle, "IF", self.nextState.ID["Instr"])

        else:
            self.nextState.ID["nop"] = True
            self.print_current_instruction(self.cycle, "IF", "nop")


        if (self.state.IF['halt'] or self.state.IF['nop']) and (self.state.ID['halt'] or self.state.ID['nop']) and (
                self.state.EX['halt'] or self.state.EX['nop']) and (self.state.MEM['halt'] or self.state.MEM['nop']) and (
                self.state.WB['halt'] or self.state.WB['nop']):
        # if self.state.IF["nop"] and self.state.ID["nop"]  and self.state.EX["nop"] and self.state.MEM["nop"] and self.state.WB["nop"]:
            # self.nextState.IF['instruction_count'] = self.state.IF['instruction_count'] + 1
            self.halted = True

            self.print_current_instruction(self.cycle, "--", "End of Simulation")

        self.myRF.output_rf(self.cycle)  # dump RF
        self.printState(self.nextState, self.cycle)  # print states after executing cycle 0, cycle 1, cycle 2 ...

        self.state = copy.deepcopy(self.nextState)
        # self.nextState = State()
        self.cycle += 1
        print(Style.RESET_ALL)


    def printState(self, state, cycle):
        print_state = ["-"*70+"\n", "State after executing cycle: " + str(cycle) + "\n"]
        print_state.extend(["IF." + key + ": " + str(val) + "\n" for key, val in state.IF.items()])
        print_state.extend(["ID." + key + ": " + str(val) + "\n" for key, val in state.ID.items()])
        print_state.extend(["EX." + key + ": " + str(val) + "\n" for key, val in state.EX.items()])
        print_state.extend(["MEM." + key + ": " + str(val) + "\n" for key, val in state.MEM.items()])
        print_state.extend(["WB." + key + ": " + str(val) + "\n" for key, val in state.WB.items()])


        if (cycle == 0):
            perm = "w"
        else:
            perm = "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(print_state)

def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    print(Style.RESET_ALL)
    exit(0)

if __name__ == "__main__":
    signal(SIGINT, handler)
    # parse arguments for input file location
    parser = argparse.ArgumentParser(description='RV32I processor')
    parser.add_argument('--iodir', default="", type=str, help='Directory containing the input files.')
    args = parser.parse_args()
    test_case_number = 1

    ioDir = os.path.abspath(args.iodir)

    print("IO Directory:", ioDir)

    imem = InsMem("Imem", ioDir)
    dmem_ss = DataMem("SS", ioDir)
    dmem_fs = DataMem("FS", ioDir)

    ssCore = SingleStageCore(ioDir, imem, dmem_ss)
    fsCore = FiveStageCore(ioDir, imem, dmem_fs)

    while True:
        if not ssCore.halted:
            ssCore.step()

        if not fsCore.halted:
            fsCore.step()

        if ssCore.halted and fsCore.halted:
            break

    # dump SS and FS data mem.
    dmem_ss.output_data_mem()
    dmem_fs.output_data_mem()


    ssCore.calculate_performance_metrics()
    fsCore.calculate_performance_metrics()

