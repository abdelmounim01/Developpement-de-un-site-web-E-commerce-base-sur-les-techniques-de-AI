import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { MayLikeComponent } from './may-like.component';

describe('MayLikeComponent', () => {
  let component: MayLikeComponent;
  let fixture: ComponentFixture<MayLikeComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ MayLikeComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(MayLikeComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
