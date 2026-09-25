set pagination off
set confirm off
catch throw bad_alloc
run
echo \n===== THROW STATE =====\n
frame 1
echo --- alloc_bytes live regs (r12=aligned_off r13=arena_this r14=base_ r15=end) ---\n
info registers r12 r13 r14 r15
echo --- arena members: base_, cap_, off_, peak_ ---\n
x/4gx $r13
frame 3
echo --- cached_launch regs ---\n
info registers rbx rbp r12 r13 r14 r15
python
import gdb
sp = int(gdb.parse_and_eval("$sp"))
gdb.write("INITLIST at sp+0xe0 (%s):\n" % hex(sp + 0xe0))
gdb.execute("x/8dw " + hex(sp + 0xe0))
end
echo \n===== BACKTRACE =====\n
bt 5
kill
