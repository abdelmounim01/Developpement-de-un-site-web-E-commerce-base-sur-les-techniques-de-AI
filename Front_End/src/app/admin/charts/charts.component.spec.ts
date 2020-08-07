import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { ChartsComponentAdmin } from './charts.component';

describe('ChartsComponent', () => {
  let component: ChartsComponentAdmin;
  let fixture: ComponentFixture<ChartsComponentAdmin>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ ChartsComponentAdmin ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(ChartsComponentAdmin);
    component = fixture.componentInstance;ChartsComponentAdmin
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
