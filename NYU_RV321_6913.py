import argparse
import os
import copy
from bitstring import BitArray

MemSize = 1000 # memory size, in reality, the memory size should be 2^32, but for this lab, for the space resaon, we keep it as this large number, but the memory is still 32-bit addressable.

class InsMem:

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

class DataMem:
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

class RegisterFile:
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

class State:
    def __init__(self):
        self.IF = {"nop": 0, "PC":0, "halt": 0}
        self.ID = {"nop": 0, "Instr": 0, "halt": 0}
        self.EX = {"nop": 0, "Read_data1": 0, "Read_data2": 0, "Rd":0, "Store_data":0, \
                   "Wrt_reg_addr": 0, "rd_mem": 0, "wrt_mem": 0,  "wrt_enable": 0, \
                    "instruction_ob": None, "halt": 0}
        self.MEM = {"nop": 0, "ALUresult": 0, "Store_data": 0,  "Wrt_reg_addr": 0, "rd_mem": 0, 
                   "wrt_mem": 0, "wrt_enable": 0, "instruction_ob": None, "halt": 0}
        self.WB = {"nop": 0, "Wrt_data": 0, "Wrt_reg_addr": 0, "wrt_enable": 0, \
                   "instruction_ob": None, "halt": 0}

    def nop_init(self):
        self.IF["nop"] = 0
        self.ID["nop"] = 1
        self.EX["nop"] = 1
        self.MEM["nop"] = 1
        self.WB["nop"] = 1

    def __str__(self):
        # DONE: update __str__ to make use of individual State objects
        return "\n\n".join([str(self.IF), str(self.ID), str(self.EX), str(self.MEM), str(self.WB)])

def cal_imm(bin_num):
    # convert the binary string to an integer
    num = int(bin_num, 2)
    # get the number of bits in the binary string
    num_bits = len(bin_num)
    # check if the number is negative
    if (num & (1 << (num_bits - 1))) != 0:
        # compute the two's complement by flipping the bits and adding 1
        num = num - (1 << num_bits)
    return num

class Instruction:
    def __init__(self, instruction:str):
        func7 = instruction[:7]
        func3 = instruction[17:20]
        opcode = instruction[25:]
        rs1 = instruction[12:17]
        rs2 = instruction[7:12]
        rd = instruction[20:25]

        if opcode == '0110011':
            if func3 == '000':
                if func7 == '0000000':
                    self.mnemonic = 'ADD'
                elif func7 == '0100000':
                    self.mnemonic = 'SUB'
            elif func3 == '100':
                self.mnemonic = 'XOR'
            elif func3 == '110':
                self.mnemonic = 'OR'
            elif func3 == '111':
                self.mnemonic = 'AND'

            self.rs1 = int(rs1,2)
            self.rs2 = int(rs2,2)
            self.rd = int(rd,2)
        elif opcode == '0010011':
            if func3 == '000': 
                self.mnemonic = 'ADDI'
            elif func3 == '100': 
                self.mnemonic =  'XORI'
            elif func3 == '110': 
                self.mnemonic =  'ORI'
            elif func3 == '111': 
                self.mnemonic = 'ANDI'

            self.imm = cal_imm(func7 + rs2)
            self.rs1 = int(rs1,2)
            self.rd = int(rd,2)
        elif opcode == '0000011':
            if func3 == '000': 
                self.mnemonic = 'LW'
            self.imm = cal_imm(func7 + rs2)
            self.rs1 = int(rs1,2)
            self.rd = int(rd,2)

        elif opcode == '1101111':
            self.mnemonic = 'JAL'
            self.imm = cal_imm( instruction[0] + instruction[12:20] + instruction[1:11] + '0')
            self.rd = int(rd, 2)

        elif opcode == '1100011':
            if func3 == '000': 
                self.mnemonic = 'BEQ'
            elif func3 == '001': 
                self.mnemonic = 'BNE'

            self.rs1 = int(rs1,2)
            self.rs2 = int(rs2,2)

            self.imm = cal_imm(instruction[0] + instruction[24] + instruction[1:7] + instruction[20:24] + '0')
            
        elif opcode == '0100011':
            self.mnemonic = 'SW'

            self.rs1 = int(rs1,2)
            self.rs2 = int(rs2,2)
            self.imm = cal_imm(func7 + rd)

        elif opcode == '1111111':
            self.mnemonic = 'HALT'

    def __str__(self):
        return self.mnemonic

class InstructionBase:
    def __init__(self, instruction, memory, registers, state,
                 nextState):
        self.instruction = instruction
        self.memory = memory
        self.registers = registers
        self.state = state
        self.nextState = nextState
        self.stages = self.memory.id

    def decode_ss(self, *args, **kwargs):
        pass

    def execute_ss(self, *args, **kwargs):
        pass

    def mem_ss(self, *args, **kwargs):
        pass

    def wb_ss(self, *args, **kwargs):
        pass

    def decode_fs(self, *args, **kwargs):
        pass

    def execute_fs(self, *args, **kwargs):
        pass

    def mem_fs(self, *args, **kwargs):

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

class InstructionRBase(InstructionBase):
    def __init__(self, instruction, memory, registers, state,
                 nextState):
        super().__init__(instruction, memory, registers, state, nextState)
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
            'wrt_enable' :1
        })

        # Stall
        if self.state.EX['Rd'] in [self.rs1, self.rs2] \
                and self.state.EX['rd_mem'] \
                and  self.rs1 != 0 and self.rs2 != 0:
            ex_state['nop'] = 1
            self.state.IF['PC'] -= 4
            self.nextState.EX = ex_state
            return

        # Forwarding
        if self.state.MEM['rd_mem'] and self.state.MEM['wrt_enable'] and \
            not self.state.MEM['wrt_mem'] and \
            self.state.MEM['Wrt_reg_addr'] == self.rs1 and self.rs1 != 0:

            ex_state['Read_data1'] = self.nextState.WB['Wrt_data']

        if self.state.MEM['rd_mem'] and self.state.MEM['wrt_enable'] and \
            not self.state.MEM['wrt_mem'] and \
            self.state.MEM['Wrt_reg_addr'] == self.rs2 and self.rs2 != 0:

            ex_state['Read_data2'] = self.nextState.WB['Wrt_data']

        if not self.state.EX['rd_mem'] and self.state.EX['wrt_enable'] and \
            not self.state.EX['wrt_mem'] and self.state.EX['Rd'] == self.rs1 and \
            self.rs1 != 0:


            ex_state['Read_data1'] = self.nextState.MEM['Store_data']

        if not self.state.EX['rd_mem'] and self.state.EX['wrt_enable'] and \
            not self.state.EX['wrt_mem'] and self.state.EX['Rd'] == self.rs2 and \
            self.rs2 != 0:


            ex_state['Read_data2'] = self.nextState.MEM['Store_data']

        self.nextState.EX = ex_state

    def execute_fs(self, *args, **kwargs):
        mem_state = State().MEM
        mem_state.update({
            'instruction_ob':self,
            'nop' :self.state.EX['nop'],
            'Wrt_reg_addr' :self.state.EX['Rd'],
            'wrt_enable' :1,
            'halt' :self.state.EX['halt']
        })
        self.nextState.MEM = mem_state

class InstructionIBase(InstructionBase):
    def __init__(self, instruction, memory, registers, state,
                 nextState):
        super().__init__(instruction, memory, registers, state, nextState)
        self.rs1 = instruction.rs1
        self.rd = instruction.rd
        self.imm = instruction.imm

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
            'wrt_enable' :1,
            'halt' :self.state.ID['halt']
        })

        # Stall
        if self.state.EX['Rd'] == self.rs1 and self.state.EX['rd_mem'] and self.rs1 != 0:
            ex_state['nop'] = 1
            self.state.IF['PC'] -= 4
            self.nextState.EX = ex_state
            return

        # Forwarding
        if self.state.MEM['rd_mem'] and self.state.MEM['wrt_enable'] and \
            not self.state.MEM['wrt_mem'] and self.state.MEM['Wrt_reg_addr'] == self.rs1 \
            and self.rs1 != 0:


            ex_state['Read_data1'] = self.nextState.WB['Wrt_data']

        if not self.state.EX['rd_mem'] and self.state.EX['wrt_enable'] and \
            not self.state.EX['wrt_mem'] and self.state.EX['Rd'] == self.rs1 and \
            self.rs1 != 0:


            ex_state["Read_data1"] = self.nextState.MEM['Store_data']

        self.nextState.EX = ex_state

    def execute_fs(self, *args, **kwargs):
        mem_state = State().MEM

        mem_state.update({
            'instruction_ob':self,
            'nop' :self.state.EX['nop'],
            'Wrt_reg_addr' :self.state.EX['Rd'],
            'wrt_enable' :1,
            'halt' :self.state.EX['halt']
        })
        self.nextState.MEM = mem_state

class InstructionSBase(InstructionBase):
    def __init__(self, instruction, memory, registers, state,
                 nextState):
        super().__init__(instruction, memory, registers, state, nextState)
        self.rs1 = instruction.rs1
        self.rs2 = instruction.rs2
        self.imm = instruction.imm

    def mem_ss(self, *args, **kwargs):
        address = kwargs['alu_result']
        data = self.registers.read_rf(self.rs2)
        self.memory.write_data_mem(address, data)

    def decode_fs(self, *args, **kwargs):
        ex_state = State().EX

        ex_state.update({
            'instruction_ob':self,
            'nop' :self.state.ID['nop'],
            'Read_data1' :self.registers.read_rf(self.rs1),
            'Read_data2' :self.imm,
            'Store_data' :self.registers.read_rf(self.rs2),
            'Rd' :self.rs2,
            'wrt_mem' :1,
            'halt':self.state.ID['halt'] 
        })
        # Stall
        if self.state.EX['Rd'] in [self.rs1, self.rs2] and \
            self.state.EX['rd_mem'] and self.rs1 != 0 and self.rs2 != 0:

            ex_state['nop'] = 1
            self.state.IF['PC'] -= 4
            self.nextState.EX = ex_state
            return

        # Forwarding
        if not self.state.EX['rd_mem'] and self.state.EX['wrt_enable'] and \
            not self.state.EX['wrt_mem'] and self.state.EX['Rd'] == self.rs1 \
            and self.rs1 != 0:


            ex_state['Read_data1'] = self.nextState.MEM['Store_data']

        if not self.state.EX['rd_mem'] and self.state.EX['wrt_enable'] and \
            not self.state.EX['wrt_mem'] and self.state.EX['Rd'] == self.rs2 and \
            self.rs2 != 0:


            ex_state['Store_data'] = self.nextState.MEM['Store_data']

        if not self.state.MEM['rd_mem'] and self.state.MEM['wrt_enable'] and \
            not self.state.MEM['wrt_mem'] and self.state.MEM['Wrt_reg_addr'] == self.rs1 and \
            self.rs1 != 0:


            ex_state['Read_data1'] = self.nextState.WB['Wrt_data']

        if not self.state.MEM['rd_mem'] and self.state.MEM['wrt_enable'] and \
            not self.state.MEM['wrt_mem'] and self.state.MEM['Wrt_reg_addr'] == self.rs2 and \
            self.rs2 != 0:


            ex_state['Store_data'] = self.nextState.WB['Wrt_data']

        self.nextState.EX = ex_state

    def execute_fs(self, *args, **kwargs):
        mem_state = State().MEM
        mem_state.update({
            'instruction_ob':self,
            'nop' :self.state.EX['nop'],
            'data_address' :self.state.EX['Read_data1'] + self.state.EX['Read_data2'],
            'Store_data' :self.state.EX['Store_data'],
            'wrt_mem' :1,
            'halt':self.state.ID['halt']
        })
        self.nextState.MEM = mem_state

    def mem_fs(self, *args, **kwargs):
        if self.state.MEM['wrt_mem']:
            self.memory.write_data_mem(self.state.MEM['data_address'], \
                                       self.state.MEM['Store_data'])
        wb_state = State().WB
        wb_state.update({
            'instruction_ob':self
        })

        self.nextState.WB = wb_state

class InstructionBBase(InstructionBase):
    def __init__(self, instruction, memory, registers, state,
                 nextState):
        super().__init__(instruction, memory, registers, state, nextState)
        self.rs1 = instruction.rs1
        self.rs2 = instruction.rs2
        self.imm = instruction.imm

    def take_branch(self, operand1, operand2):
        pass

    def execute_ss(self, *args, **kwargs):
        pass

    def execute_fs(self, *args, **kwargs):
        mem_state = State().MEM
        mem_state['instruction_ob'] = self
        mem_state['nop'] = 1
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
            self.nextState.ID['nop'] = 1
            self.state.IF['nop'] = 1
        ex_state['nop'] = 1

        self.nextState.EX = ex_state

class InstructionJBase(InstructionBase):
    def __init__(self, instruction, memory, registers, state,nextState):
        super().__init__(instruction, memory, registers, state, nextState)
        self.rd = instruction.rd
        self.imm = instruction.imm

    def execute_ss(self, *args, **kwargs):
        pass

    def decode_fs(self, *args, **kwargs):
        ex_state = State().EX
        ex_state.update({
            'instruction_ob' :self,
            'Store_data' :self.state.IF['PC'],
            'Rd' :self.rd,
            'wrt_enable' :1
        })

        self.nextState.IF['PC'] = self.state.IF['PC'] + self.imm - 4
        self.nextState.ID['nop'] = 1
        self.state.IF['nop'] = 1

        self.nextState.EX = ex_state

    def execute_fs(self, *args, **kwargs):

        mem_state = State().MEM
        mem_state.update({
            'instruction_ob':self,
            'Store_data' :self.state.EX['Store_data'],
            'Wrt_reg_addr' :self.rd,
            'wrt_enable' :1
        })

        self.nextState.MEM = mem_state

class ADD(InstructionRBase):
    def __init__(self, instruction, memory, registers, state,
                 nextState):
        super().__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) + self.registers.read_rf(self.rs2)

    def execute_fs(self, *args, **kwargs):
        super().execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] + self.state.EX['Read_data2']
class SUB(InstructionRBase):

    def __init__(self, instruction, memory, registers, state, nextState):
        super().__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) - self.registers.read_rf(self.rs2)

    def execute_fs(self, *args, **kwargs):
        super().execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] - self.state.EX['Read_data2']

class XOR(InstructionRBase):

    def __init__(self, instruction, memory, registers, state, nextState):
        super().__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) ^ self.registers.read_rf(self.rs2)

    def execute_fs(self, *args, **kwargs):
        super().execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] ^ self.state.EX['Read_data2']

class OR(InstructionRBase):

    def __init__(self, instruction, memory, registers, state,
                 nextState):
        super().__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) | self.registers.read_rf(self.rs2)

    def execute_fs(self, *args, **kwargs):
        super().execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] | self.state.EX['Read_data2']

class AND(InstructionRBase):
    def __init__(self, instruction, memory, registers, state,
                 nextState):
        super().__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) & self.registers.read_rf(self.rs2)

    def execute_fs(self, *args, **kwargs):
        super().execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] & self.state.EX['Read_data2']

class ADDI(InstructionIBase):
    def __init__(self, instruction, memory, registers, state,
                 nextState):
        super().__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) + self.imm

    def execute_fs(self, *args, **kwargs):
        super().execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] + self.state.EX['Read_data2']

class XORI(InstructionIBase):
    def __init__(self, instruction, memory, registers, state,
                 nextState):
        super().__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) ^ self.imm

    def execute_fs(self, *args, **kwargs):
        super().execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] ^ self.state.EX['Read_data2']

class ORI(InstructionIBase):

    def __init__(self, instruction, memory, registers, state,
                 nextState):
        super().__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) | self.imm

    def execute_fs(self, *args, **kwargs):
        super().execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] | self.state.EX['Read_data2']

class ANDI(InstructionIBase):
    def __init__(self, instruction, memory, registers, state,
                 nextState):
        super().__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) & self.imm

    def execute_fs(self, *args, **kwargs):
        super().execute_fs()
        self.nextState.MEM['Store_data'] = self.state.EX['Read_data1'] & self.state.EX['Read_data2']

class LW(InstructionIBase):
    def __init__(self, instruction, memory, registers, state,
                 nextState):
        super().__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) + self.imm

    def mem_ss(self, *args, **kwargs):
        address = kwargs['alu_result']
        return self.memory.read_data(address)

    def wb_ss(self, *args, **kwargs):
        data = kwargs['mem_result']
        return self.registers.write_rf(self.rd, data)

    def decode_fs(self, *args, **kwargs):
        super().decode_fs()
        self.nextState.EX['rd_mem'] = 1

    def execute_fs(self, *args, **kwargs):
        super().execute_fs()
        self.nextState.MEM.update({
            'data_address': self.state.EX['Read_data1'] + self.state.EX['Read_data2'],
            'rd_mem' : 1
        })

    def mem_fs(self, *args, **kwargs):
        super().mem_fs(*args, **kwargs)
        if self.state.MEM['rd_mem']:
            self.nextState.WB['Wrt_data'] = self.memory.read_data(
                self.state.MEM['data_address']
            )

class SW(InstructionSBase):
    def __init__(self, instruction, memory, registers, state,
                 nextState):
        super().__init__(instruction, memory, registers, state, nextState)

    def execute_ss(self, *args, **kwargs):
        return self.registers.read_rf(self.rs1) + self.imm

class BEQ(InstructionBBase):

    def __init__(self, instruction, memory, registers, state,
                 nextState):
        super().__init__(instruction, memory, registers, state, nextState)

    def take_branch(self, operand1, operand2):
        return operand1 == operand2

class BNE(InstructionBBase):

    def __init__(self, instruction, memory, registers, state,
                 nextState):
        super().__init__(instruction, memory, registers, state, nextState)

    def take_branch(self, operand1, operand2):
        return operand1 != operand2

class JAL(InstructionJBase):

    def __init__(self, instruction, memory, registers, state,
                 nextState):
        super().__init__(instruction, memory, registers, state, nextState)

class ADDERBTYPE:
    def __init__(self, instruction, state, registers):
        self.instruction = instruction
        self.state = state
        self.registers = registers
        self.rs1 = instruction.rs1
        self.rs2 = instruction.rs2
        self.imm = instruction.imm

    def get_pc(self, *args, **kwargs):
        if self.instruction.mnemonic == 'BEQ':
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
    def __init__(self, instruction, state, registers):
        self.instruction = instruction
        self.state = state
        self.registers = registers
        self.rd = instruction.rd
        self.imm = instruction.imm

    def get_pc(self, *args, **kwargs):
        self.registers.write_rf(self.rd, self.state.IF['PC'] + 4)
        return self.state.IF['PC'] + self.imm

def get_instruction_class(mnemonic):
    try:       
        cls = eval(mnemonic.upper())
        return cls
    except AttributeError as e:
        raise Exception("Invalid Instruction")

class Core:
    def __init__(self, ioDir, imem, dmem):
        self.myRF = RegisterFile(ioDir)
        self.cycle = 0
        self.halted = 0
        self.ioDir = ioDir
        self.state = State()
        self.state.nop_init()
        self.nextState = State()
        self.nextState.nop_init()
        self.ext_imem: InsMem = imem
        self.ext_dmem = dmem

    def calculate_performance_metrics(self, mode='w'):
        if self.stage =='SS':
            #cpi will be always 1 for Single Stage Core
            cpi =1
            ipc =1
            type = 'Single Stage'
        else:
            cpi = round(float(self.cycle) / (len(self.ext_imem.IMem)/4),6)
            ipc = round(1 / cpi,6)
            type = 'Five Stage'
        result_format = f"{self.ioDir[:-3]}\n" \
                        f"{type} Core Performance Metrics{'-'*30}\n" \
                        f"Number of cycles taken: {self.cycle}\n" \
                        f"Cycles per instruction: {cpi}\n" \
                        f"Instructions per cycle: {ipc}\n"

        print(self.ioDir[:-3] + "PerformanceMetrics_Result.txt")
        with open(self.ioDir[:-3] + "PerformanceMetrics_Result.txt", mode) as file:
            file.write(result_format)

class SingleStageCore(Core):
    def __init__(self, ioDir, imem, dmem):
        super().__init__(ioDir + "/SS_", imem, dmem)
        self.opFilePath = ioDir + "/StateResult_SS.txt"
        self.stage = "SS"


    def step(self):
        # IF
        instruction_bytes = self.ext_imem.read_instr(self.state.IF['PC'])

        if instruction_bytes == "1" * 32:
            self.nextState.IF['nop'] = 1
        else:
            self.nextState.IF['PC'] += 4

        try:
            # ID
            instruction = Instruction(instruction_bytes)

            if instruction.mnemonic =='HALT':
                pass
            
            elif instruction.mnemonic in ['BEQ', 'BNE']:
                self.nextState.IF['PC'] = ADDERBTYPE(instruction, self.state, self.myRF).get_pc()
            elif instruction.mnemonic == 'JAL':
                self.nextState.IF['PC'] = ADDERJTYPE(instruction, self.state, self.myRF).get_pc()
            else:
                instruction_ob = get_instruction_class(instruction.mnemonic)(instruction,
                                                                            self.ext_dmem, self.myRF,
                                                                            self.state,
                                                                            self.nextState)
                # Ex
                alu_result = instruction_ob.execute()
                # Load/Store (MEM)
                mem_result = instruction_ob.mem(alu_result=alu_result)
                # WB
                instruction_ob.wb(mem_result=mem_result, alu_result=alu_result)
        except Exception as e:
            raise Exception("Invalid Instruction to Decode")

        if self.state.IF['nop']:
            self.halted = 1

        self.myRF.output_rf(self.cycle)  # dump RF
        self.printState(self.nextState, self.cycle)  # print states after executing cycle 0, cycle 1, cycle 2 ...

        # The end of the cycle and updates the current state with the values calculated in this cycle
        self.state = copy.deepcopy(self.nextState)
        self.cycle += 1

    def printState(self, state, cycle):
        printstate = ["State after executing cycle:\t" + str(cycle) + "\n"]
        printstate.append(f"IF.PC:\t" + str(state.IF['PC']) + "\n")
        printstate.append(f"IF.nop:\t" + str(state.IF['nop']) + "\n")

        if (cycle == 0):perm = "w"
        else: perm = "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)

class FiveStageCore(Core):
    def __init__(self, ioDir, imem, dmem):
        super().__init__(ioDir + "/FS_", imem, dmem)
        self.opFilePath = ioDir + "/StateResult_FS.txt"
        self.stage = "FS"

    def step(self):
        # Your implementation
        # --------------------- WB stage ----------------------
        if not self.state.WB["nop"]:

            self.state, self.nextState, self.ext_dmem, self.myRF, _ = self.state.WB["instruction_ob"].wb(
                state=self.state,
                nextState=self.nextState,
                registers=self.myRF,
                memory=self.ext_dmem)

        # --------------------- MEM stage ---------------------
        if not self.state.MEM["nop"]:

            self.state, self.nextState, self.ext_dmem, self.myRF, _ = self.state.MEM["instruction_ob"].mem(
                state=self.state,
                nextState=self.nextState,
                registers=self.myRF,
                memory=self.ext_dmem)
        else:

            self.nextState.WB["nop"] = 1

        # --------------------- EX stage ----------------------
        if not self.state.EX["nop"]:
            self.state, self.nextState, self.ext_dmem, self.myRF, _ = self.state.EX["instruction_ob"].execute(
                state=self.state, nextState=self.nextState, registers=self.myRF, memory=self.ext_dmem)
        else:
            self.nextState.MEM["nop"] = 1

        # --------------------- ID stage ----------------------
        if not self.state.ID["nop"]:
            try:
                instruction = Instruction(self.state.ID['Instr'])
                if instruction.mnemonic =='HALT':
                    self.nextState.ID["halt"] = 1
                    
                else:
                    instruction_ob  = get_instruction_class(instruction.mnemonic)(instruction,
                                                                                    self.ext_dmem,
                                                                                    self.myRF,
                                                                                    self.state,
                                                                                    self.nextState)
                
                    self.state, self.nextState, self.ext_dmem, self.myRF, _ = instruction_ob.decode(state=self.state,
                                                                                                    nextState=self.nextState,
                                                                                                    registers=self.myRF,
                                                                                                    memory=self.ext_dmem)
                    
            except Exception as e:
                raise Exception("Invalid Instruction to Decode")
        else:
            self.nextState.EX["nop"] = 1

        # --------------------- IF stage ----------------------
        if not self.state.IF["nop"]:
            self.nextState.ID['Instr'] = self.ext_imem.read_instr(self.state.IF["PC"])
            self.nextState.ID["nop"] = 0
            if self.nextState.ID['Instr'] == "1" * 32:
                self.nextState.ID["nop"] = 1
                self.nextState.IF["nop"] = 1
            else:
                self.nextState.IF["PC"] = self.state.IF["PC"] + 4

        else:
            self.nextState.ID["nop"] = 1

        if (self.state.IF['halt'] or self.state.IF['nop']) and (self.state.ID['halt'] or self.state.ID['nop']) and (
                self.state.EX['halt'] or self.state.EX['nop']) and (self.state.MEM['halt'] or self.state.MEM['nop']) and (
                self.state.WB['halt'] or self.state.WB['nop']):
            self.halted = 1

        self.myRF.output_rf(self.cycle)  # dump RF
        self.printState(self.nextState, self.cycle)  # print states after executing cycle 0, cycle 1, cycle 2 ...

        self.state = copy.deepcopy(self.nextState)
        self.cycle += 1

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

if __name__ == "__main__":
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

    while 1:
        if not ssCore.halted:
            ssCore.step()

        if not fsCore.halted:
            fsCore.step()

        if ssCore.halted and fsCore.halted:
            break

    # dump SS and FS data mem.
    dmem_ss.output_data_mem()
    dmem_fs.output_data_mem()

    ssCore.calculate_performance_metrics('w')
    fsCore.calculate_performance_metrics('a')