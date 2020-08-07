import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { MetausersComponent } from './metausers.component';

describe('MetausersComponent', () => {
  let component: MetausersComponent;
  let fixture: ComponentFixture<MetausersComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ MetausersComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(MetausersComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
