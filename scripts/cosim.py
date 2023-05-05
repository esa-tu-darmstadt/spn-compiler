"""
Inspired by loopback.mlir
"""

#!/usr/bin/python3

# Current CIRCT version: bd11d23489a1dd7f64a0e9028e63c4e0b79e8cb7

import binascii
import random
import esi_cosim
import struct


EP_NAME = 'top.wrapper_instance.ESIEndpoint_instance.MyEndpoint'


class CosimTester(esi_cosim.CosimBase):
  """Provides methods to test the loopback simulations."""

  def test_list(self):
    ifaces = self.cosim.list().wait().ifaces
    assert len(ifaces) > 0

  def test_open_close(self):
    ifaces = self.cosim.list().wait().ifaces
    openResp = self.cosim.open(ifaces[0]).wait()
    assert openResp.iface is not None
    ep = openResp.iface
    ep.close().wait()

  def test_send_receive(self):
    ep = self.openEP(EP_NAME)

    ep.send(
      self.schema.Struct17804976951450814598.new_message(
        last=True,
        v0=0,
        v1=0,
        v2=0,
        v3=0,
        v4=0
      )
    )

    got = self.readMsg(ep, self.schema.Struct15544314350482218497)
    value = struct.unpack('d', got.data.to_bytes(8))
    print(f'got: {value}')


if __name__ == "__main__":
  import os
  import sys

  # sys.argv[1] seems to be enough for a host string
  host_string = sys.argv[1]
  rpc = CosimTester(sys.argv[2], f"{host_string}")
  print(rpc.list())

  rpc.test_list()
  rpc.test_open_close()

  rpc.test_send_receive()
  rpc.test_send_receive()
  rpc.test_send_receive()
  rpc.test_send_receive()
  rpc.test_send_receive()
  rpc.test_send_receive()
  rpc.test_send_receive()
  rpc.test_send_receive()
  rpc.test_send_receive()
  rpc.test_send_receive()

  print('cosim done')
