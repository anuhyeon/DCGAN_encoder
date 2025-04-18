print(fake_data.requires_grad)  # True
print(fake_data.grad_fn)        # <Some function> — 연산 이력 있음

for p in netG.parameters():
    print(p.grad)  # None ← 아직 backward 안 했으면 없음

errG.backward()

for p in netG.parameters():
    print(p.grad)  # 이제는 gradient가 있음
